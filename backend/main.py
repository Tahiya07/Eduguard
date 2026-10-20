from __future__ import annotations
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from .auth import create_token, current_session, require_teacher
from .config import settings
from .schemas import BloomRequest, ChatRequest, IndexTextRequest, LoginRequest, ModerationReviewRequest, TargetRewriteRequest, TextRequest
from .service import service
from operation_control import OperationCancelled, cancel_operation, check_cancelled, operation_context


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Uvicorn has already bound its listening socket before lifespan startup.
    # Start the local GGUF load in a background thread so model initialization
    # does not hold up ASGI startup or block health/status requests. Set the
    # flag to 0 only when intentionally preferring first-request lazy loading.
    if os.getenv("EDUGUARD_WARMUP_ON_STARTUP", "1").strip().lower() in {"1", "true", "yes"}:
        threading.Thread(target=service.warmup, name="eduguard-warmup", daemon=True).start()
    yield


app = FastAPI(title="Framework Academic API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins(),
    # Development may choose another local port when 3000 is occupied. Permit
    # only loopback origins at any port so browser preflights still succeed.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
def run(fn):
    try:
        check_cancelled()
        result = fn()
        check_cancelled()
        return result
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except OperationCancelled as e:
        raise HTTPException(409, str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))

def run_operation(operation_id: str, fn):
    if not operation_id:
        return run(fn)
    if len(operation_id) > 128:
        raise HTTPException(400, "Invalid X-Operation-ID header.")
    with operation_context(operation_id):
        return run(fn)

@app.post("/operations/cancel")
def cancel_running_operation(operation_id: str, session: dict = Depends(current_session)):
    return {"cancelled": cancel_operation(operation_id), "operation_id": operation_id}

@app.get("/health")
def health(): return {"status": "ok", "offline_mode": settings.offline_mode}
@app.get("/models/status")
def models_status(session: dict = Depends(current_session)): return service.status()
@app.post("/models/load-bloom")
def load_bloom(session: dict = Depends(require_teacher)):
    # FastAPI runs sync handlers in a worker thread, so model loading does not block health checks.
    return run(service.load_bloom)
@app.post("/auth/session")
def login(body: LoginRequest):
    expected = settings.teacher_access_code if body.role == "teacher" else settings.student_access_code
    if body.access_code != expected: raise HTTPException(401, "Invalid access code.")
    return {"access_token": create_token(body.role), "role": body.role}
@app.post("/documents/index")
def index_text(body: IndexTextRequest, session: dict = Depends(current_session), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.index_text(session["sid"], session["role"], body.text, body.name, body.scope, body.content_type))
@app.post("/documents/upload")
async def upload(file: UploadFile = File(...), scope: str = Form("public"), content_type: str = Form("study_material"), session: dict = Depends(current_session), operation_id: str = Header(default="", alias="X-Operation-ID")):
    payload = await file.read()
    # PDF parsing, OCR, embedding, and FAISS indexing are CPU-bound; keep them
    # off the ASGI event loop so health/status and other requests remain responsive.
    return await run_in_threadpool(
        lambda: run_operation(operation_id, lambda: service.ingest_file(
            session["sid"], session["role"], payload, file.filename or "upload", scope, content_type
        ))
    )
# Bloom inference is a teacher-side capability. Keep the legacy route but
# protect it identically to the explicit teacher endpoint.
@app.post("/bloom/classify")
def classify(body: BloomRequest, session: dict = Depends(require_teacher), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.classify(body.question))
@app.post("/teacher/exam/classify")
def classify_exam(body: BloomRequest, session: dict = Depends(require_teacher), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.classify(body.question))
@app.post("/teacher/exam/moderate")
def moderate_exam(body: BloomRequest, session: dict = Depends(require_teacher), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.moderate_exam_question(session["sid"], body.question))
@app.post("/teacher/exam/linguistic-moderate")
def linguistic_moderate(body: BloomRequest, session: dict = Depends(require_teacher), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.linguistic_moderate_question(session["sid"], body.question))
@app.post("/teacher/exam/target-rewrite")
def target_rewrite(body: TargetRewriteRequest, session: dict = Depends(require_teacher), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.target_level_rewrite(session["sid"], body.question, body.target_level))
@app.post("/teacher/exam/review")
def review_exam(body: ModerationReviewRequest, session: dict = Depends(require_teacher)): return service.record_moderation_review(session["sid"], body.question, body.decision, body.notes)
@app.post("/qa")
def qa(body: TextRequest, session: dict = Depends(current_session), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.answer(session["sid"], session["role"], body.question, body.scope, body.top_k))
@app.post("/chat")
def chat(body: ChatRequest, session: dict = Depends(current_session), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.student_chat(session["sid"], session["role"], body.question, body.scope, body.top_k, [turn.model_dump() for turn in body.history], summary=body.summary))
@app.post("/summarize")
def summarize(body: TextRequest, session: dict = Depends(current_session), operation_id: str = Header(default="", alias="X-Operation-ID")): return run_operation(operation_id, lambda: service.answer(session["sid"], session["role"], body.question, body.scope, body.top_k, summary=True))
@app.post("/retrieval/search")
def search(body: TextRequest, session: dict = Depends(current_session)):
    return run(lambda: {"sources": service.answer(session["sid"], session["role"], body.question, body.scope, body.top_k)["sources"]})
