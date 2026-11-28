"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ComponentType, KeyboardEvent } from "react";
import VideoHero from "./video-hero";
import type { HeroClip } from "./video-hero";
import { clearSession } from "./session";
import ErrorModal, { errorTitle } from "./error-modal";
import { type OperationProgress, useOperationProgress } from "./operation-progress";
import FilePicker from "./file-picker";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
  retries = 12,
): Promise<Response> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(input, init);

      // Backend is alive. Return the response normally.
      return response;
    } catch (error) {
      lastError = error;
      if (init?.signal?.aborted || (error instanceof DOMException && error.name === "AbortError")) throw error;

      if (attempt === retries) {
        throw lastError;
      }

      // Backend may still be initializing. Wait progressively,
      // but never block the UI itself.
      await new Promise<void>((resolve, reject) => {
        const signal = init?.signal;
        const timeout = window.setTimeout(() => {
          signal?.removeEventListener("abort", abort);
          resolve();
        }, Math.min(1000 + attempt * 500, 3000));
        const abort = () => {
          window.clearTimeout(timeout);
          reject(new DOMException("The operation was cancelled.", "AbortError"));
        };
        if (signal?.aborted) abort();
        else signal?.addEventListener("abort", abort, { once: true });
      });
    }
  }

  throw lastError;
}

function createOperation() {
  return { id: crypto.randomUUID(), controller: new AbortController() };
}

function cancelBackendOperation(operation: { id: string; controller: AbortController } | null, token: string) {
  if (!operation) return;
  operation.controller.abort();
  void fetch(`${API}/operations/cancel?operation_id=${encodeURIComponent(operation.id)}`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  }).catch(() => {
    // The local operation may have already completed; closing the UI is still immediate.
  });
}

function useElapsed(active: boolean) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!active) {
      setElapsed(0);
      return;
    }

    const start = Date.now();
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 250);

    return () => window.clearInterval(id);
  }, [active]);

  return elapsed;
}

type Bloom = { level: string; confidence: number };
type Moderation = {
  bloom_level: string;
  target_higher_level: string;
  reason: string;
  higher_level_rewrite: string;
};

type LoadingProps = {
  label: string;
  hint?: string;
  success?: boolean;
  successMessage?: string;
  finalElapsed?: number;
  showGame?: boolean;
  onCancel?: () => void;
  progress?: OperationProgress | null;
};

type Action = "classify" | "moderate" | "linguistic" | "target-rewrite" | "qa" | "summary";
type ToolId = "classify" | "rewrite" | "linguistic" | "summary" | "qa";

const BLOOM_LEVELS = [
  { level: "Remember", code: "C1", hint: "Recall facts and terms" },
  { level: "Understand", code: "C2", hint: "Explain ideas in own words" },
  { level: "Apply", code: "C3", hint: "Use knowledge in a new case" },
  { level: "Analyze", code: "C4", hint: "Break down, compare, relate" },
  { level: "Evaluate", code: "C5", hint: "Judge and justify a position" },
  { level: "Create", code: "C6", hint: "Design or produce something new" },
] as const;

const TEACHER_CLIPS: HeroClip[] = [
  { src: "/teacher-moderation-workflow.mp4", title: "Moderation workflow", caption: "Check a level, rewrite it, confirm the result." },
  { src: "/mechanism-teacher-workflow.mp4", title: "Teacher workflow", caption: "How the tools fit into exam preparation." },
  { src: "/mechanism-privacy-guard.mp4", title: "Privacy guard", caption: "Protected material stays with teachers." },
  { src: "/mechanism-local-university.mp4", title: "Runs locally", caption: "Every model runs on this computer, offline." },
];

function scrollToId(id: string, focusId?: string) {
  const el = document.getElementById(id);
  if (el instanceof HTMLDetailsElement) el.open = true;
  el?.scrollIntoView({ behavior: "smooth", block: "start" });
  if (focusId) window.setTimeout(() => document.getElementById(focusId)?.focus({ preventScroll: true }), 350);
}

// One tool = one clear job, one input meaning, one primary button.
const TOOLS: {
  id: ToolId;
  label: string;
  action: Action;
  needsClassifier: boolean;
  intro: string;
  inputLabel: string;
  placeholder: string;
  hint: string;
}[] = [
  {
    id: "classify",
    label: "Check Bloom level",
    action: "classify",
    needsClassifier: true,
    intro: "Find out which Bloom level a question tests. You can then confirm or correct the result.",
    inputLabel: "Exam or assessment question",
    placeholder: "e.g. Explain how a deadlock can occur between two processes.",
    hint: "Enter one question at a time.",
  },
  {
    id: "rewrite",
    label: "Rewrite to a level",
    action: "target-rewrite",
    needsClassifier: false,
    intro: "Rewrite a question so it tests a different Bloom level while keeping the same topic.",
    inputLabel: "Question to rewrite",
    placeholder: "e.g. List the seven layers of the OSI model.",
    hint: "Then pick the level you want below.",
  },
  {
    id: "linguistic",
    label: "Improve wording",
    action: "linguistic",
    needsClassifier: true,
    intro: "Make a question clearer and grammatically correct without changing what it asks.",
    inputLabel: "Question to improve",
    placeholder: "e.g. explain how do a stack and queue differs in there order of removing items",
    hint: "The meaning and the Bloom level stay the same.",
  },
  {
    id: "summary",
    label: "Summarize text",
    action: "summary",
    needsClassifier: false,
    intro: "Get a short, faithful summary of academic text such as lecture notes or a textbook section.",
    inputLabel: "Text to summarize",
    placeholder: "Paste a passage, lecture notes or a textbook section…",
    hint: "To summarize a whole file instead, add it under “Reference material” above and type “Summarize the document”.",
  },
  {
    id: "qa",
    label: "Ask questions",
    action: "qa",
    needsClassifier: true,
    intro: "Ask a subject question and get a concise answer from the local model.",
    inputLabel: "Your question",
    placeholder: "e.g. What is the purpose of normalization in database design?",
    hint: "Works without any uploaded material. Answers can be wrong, so check them before use.",
  },
];

function TeacherStudio({ token, LoadingComponent }: { token: string; LoadingComponent: ComponentType<LoadingProps> }) {
  const [q, setQ] = useState("");
  const [tool, setTool] = useState<ToolId>("classify");
  const [bloom, setBloom] = useState<Bloom | null>(null);
  const [mod, setMod] = useState<Moderation | null>(null);
  const [linguisticMod, setLinguisticMod] = useState<{ original: string; improved: string } | null>(null);
  const [targetRewrite, setTargetRewrite] = useState<{
    original: string;
    rewritten: string;
    target_level: string;
    verified_level?: string | null;
    attempts?: number;
  } | null>(null);
  const [targetLevel, setTargetLevel] = useState("Analyze");
  // Teacher-confirmed Bloom level, saved on this machine for federated training rounds.
  const [confirmLevel, setConfirmLevel] = useState("");
  const [labelNote, setLabelNote] = useState("");
  const [labelSaving, setLabelSaving] = useState(false);
  const [answer, setAnswer] = useState("");
  const [summary, setSummary] = useState("");
  const [busy, setBusy] = useState("");
  const operationController = useRef<ReturnType<typeof createOperation> | null>(null);
  const currentOperationId = useCallback(() => operationController.current?.id, []);
  const progress = useOperationProgress(API, token, currentOperationId, !!busy);
  const resultsRef = useRef<HTMLElement>(null);
  const cancelOperation = useCallback(() => {
    const active = operationController.current;
    operationController.current = null;
    cancelBackendOperation(active, token);
    setBusy("");
    setSuccess({ show: false, message: "", elapsed: 0 });
  }, [token]);
  const [error, setError] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [materials, setMaterials] = useState<{ id: string; name: string }[]>([]);
  const [success, setSuccess] = useState<{ show: boolean; message: string; elapsed: number }>({ show: false, message: "", elapsed: 0 });
  const operationElapsed = useElapsed(!!busy);
  const [classifierLoaded, setClassifierLoaded] = useState(false);
  const [classifierFailed, setClassifierFailed] = useState(false);
  const [classifierLoadAttempt, setClassifierLoadAttempt] = useState(0);

  async function loadBloomClassifier() {
    // Load silently in the background; do not replace or block the teacher UI.
    setClassifierFailed(false);
    try {
      const response = await apiFetch(`${API}/models/load-bloom`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.detail || "The teacher Bloom classifier could not be loaded.");
      setClassifierLoaded(true);
    } catch (e) {
      // Surface only failures so the teacher can retry; successful loading is silent.
      setClassifierFailed(true);
      setError(String(e).replace("Error: ", ""));
    }
  }

  // Start loading only after the authenticated teacher workspace has mounted.
  useEffect(() => {
    if (!classifierLoaded) void loadBloomClassifier();
    // Initial mount and explicit retry are the only triggers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classifierLoadAttempt, token]);

  function clearOutputs() {
    setBloom(null);
    setConfirmLevel("");
    setLabelNote("");
    setMod(null);
    setLinguisticMod(null);
    setTargetRewrite(null);
    setAnswer("");
    setSummary("");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
  }

  // Previous results describe the previous text, so clear them when the text changes.
  const handleQuestionChange = (newQuestion: string) => {
    setQ(newQuestion);
    clearOutputs();
  };

  const handleTargetLevelSelect = (level: string) => {
    setTargetLevel(level);
    setTargetRewrite(null);
    setError("");
  };

  async function req(path: string, body: object) {
    const active = operationController.current;
    const r = await apiFetch(`${API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
        ...(active ? { "X-Operation-ID": active.id } : {}),
      },
      body: JSON.stringify(body),
      signal: active?.controller.signal,
    });
    const d = await r.json();
    if (r.status === 401) {
      clearSession();  // returns to the access-code screen
      throw Error("Your session expired. Please enter your access code again.");
    }
    if (!r.ok) throw Error(d.detail || "The local service could not complete this request.");
    return d;
  }

  async function saveConfirmedLevel(predicted: string) {
    const chosen = confirmLevel || predicted;
    if (!q.trim() || !chosen) return;
    setLabelSaving(true);
    setLabelNote("");
    try {
      const r = await apiFetch(`${API}/teacher/exam/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          question: q,
          decision: "approved",
          bloom_level: chosen,
          predicted_level: predicted,
          predicted_confidence: bloom?.confidence ?? null,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw Error(d.detail || "Could not save the confirmed level.");
      setLabelNote(
        d.saved_for_federated_training
          ? `Saved on this computer only (${d.local_label_count} confirmed questions). Used in the next local training round.`
          : "Review recorded. Label collection is turned off on this computer.",
      );
    } catch (e) {
      setLabelNote(e instanceof Error ? e.message : "Could not save the confirmed level.");
    } finally {
      setLabelSaving(false);
    }
  }

  async function run(action: Action, explicitTargetLevel?: string) {
    // Target rewrites no longer use classifier re-scoring, so they can run while the classifier loads.
    if (!q.trim() || (!classifierLoaded && action !== "summary" && action !== "target-rewrite")) return;

    clearOutputs();

    const labels = {
      classify: "Classifying the question",
      moderate: "Preparing moderation guidance",
      linguistic: "Improving question quality",
      "target-rewrite": "Rewriting to target level",
      qa: "Loading the local model and writing a reply",
      summary: "Preparing a local summary",
    } as const;

    const successMessages = {
      classify: "Bloom level classified",
      moderate: "Moderated and rewritten",
      linguistic: "Question improved",
      "target-rewrite": "Question rewritten",
      qa: "Answered the question",
      summary: "Summarized the content",
    } as const;

    const operation = createOperation();
    operationController.current = operation;
    setBusy(labels[action]);

    try {
      if (action === "classify") {
        const d = await req("/teacher/exam/classify", { question: q });
        setBloom({ level: d.level, confidence: d.confidence });
        setSuccess({ show: true, message: successMessages.classify, elapsed: operationElapsed });
      }
      if (action === "moderate") {
        const d = await req("/teacher/exam/moderate", { question: q });
        setBloom({ level: d.bloom.level, confidence: d.bloom.confidence });
        setMod(d.moderation);
        setSuccess({ show: true, message: successMessages.moderate, elapsed: operationElapsed });
      }
      if (action === "linguistic") {
        const d = await req("/teacher/exam/linguistic-moderate", { question: q });
        setLinguisticMod({ original: d.original, improved: d.improved });
        setSuccess({ show: true, message: successMessages.linguistic, elapsed: operationElapsed });
      }
      if (action === "target-rewrite") {
        const effectiveTargetLevel = explicitTargetLevel || targetLevel;
        const d = await req("/teacher/exam/target-rewrite", { question: q, target_level: effectiveTargetLevel });

        if (!d.rewritten) {
          setError(d.error || "Could not generate a rewrite.");
          setTargetRewrite(null);
        } else {
          setTargetRewrite({
            original: d.original,
            rewritten: d.rewritten,
            target_level: d.target_level,
            verified_level: d.verified_level ?? null,
            attempts: d.attempts ?? 1,
          });
          setSuccess({ show: true, message: successMessages["target-rewrite"], elapsed: operationElapsed });
        }
      }
      if (action === "qa") {
        const d = await req("/qa", { question: q, scope: "public", top_k: 4 });
        setAnswer(d.answer);
        setSuccess({ show: true, message: successMessages.qa, elapsed: operationElapsed });
      }
      if (action === "summary") {
        const d = await req("/summarize", { question: q, scope: "public", top_k: 4 });
        setSummary(d.answer);
        setSuccess({ show: true, message: successMessages.summary, elapsed: operationElapsed });
      }
      // Bring the result into view on smaller screens.
      window.setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) {
      if (operationController.current === operation && !(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      const finalElapsed = operationElapsed;
      if (stillCurrent) setBusy("");
      if (stillCurrent && successMessages[action] && !operation.controller.signal.aborted) {
        setSuccess(prev => ({ ...prev, elapsed: finalElapsed }));
        setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 2000);
      }
    }
  }

  async function clearNotes() {
    setError("");
    try {
      const r = await apiFetch(`${API}/documents/clear?scope=public`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (r.status === 401) {
        clearSession();
        throw Error("Your session expired. Please enter your access code again.");
      }
      if (!r.ok) throw Error("Could not remove the material. Please try again.");
      setMaterials([]);
      setSuccess({ show: true, message: "Material removed", elapsed: 0 });
      setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 1500);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function index() {
    if (!text.trim()) return;
    const operation = createOperation();
    operationController.current = operation;
    setBusy("Indexing your material");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
    try {
      await req("/documents/index", {
        text,
        name: "study-notes",
        scope: "public",
        content_type: "study_material",
      });
      setMaterials((m) => [...m, { id: crypto.randomUUID(), name: `Pasted text (${text.trim().split(/\s+/).length} words)` }]);
      setText("");
      setSuccess({ show: true, message: "Indexing complete", elapsed: operationElapsed });
    } catch (e) {
      if (operationController.current === operation && !(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      const finalElapsed = operationElapsed;
      if (stillCurrent) {
        setBusy("");
        if (!operation.controller.signal.aborted) setSuccess(prev => ({ ...prev, elapsed: finalElapsed }));
        setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 2000);
      }
    }
  }

  async function upload() {
    if (!file) return;
    const operation = createOperation();
    operationController.current = operation;
    setBusy("Uploading and indexing your file");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
    try {
      const f = new FormData();
      f.set("file", file);
      f.set("scope", "public");
      f.set("content_type", "study_material");
      const r = await apiFetch(`${API}/documents/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "X-Operation-ID": operation.id },
        body: f,
        signal: operation.controller.signal,
      });
      if (r.status === 401) {
        clearSession();
        throw Error("Your session expired. Please enter your access code again.");
      }
      if (!r.ok) {
        // Show the service's own reason (e.g. OCR missing) instead of a generic message.
        const detail = await r.json().then((d) => d?.detail).catch(() => null);
        throw Error(detail || "Upload failed. Supported files: PDF, TXT, MD, CSV, JPG and PNG.");
      }
      setMaterials((m) => [...m, { id: crypto.randomUUID(), name: file.name }]);
      setFile(null);
      setSuccess({ show: true, message: "Indexing complete", elapsed: operationElapsed });
    } catch (e) {
      if (operationController.current === operation && !(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      const finalElapsed = operationElapsed;
      if (stillCurrent) {
        setBusy("");
        if (!operation.controller.signal.aborted) setSuccess(prev => ({ ...prev, elapsed: finalElapsed }));
        setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 2000);
      }
    }
  }

  const activeTool = TOOLS.find((t) => t.id === tool) ?? TOOLS[0];
  const waitingForClassifier = activeTool.needsClassifier && !classifierLoaded;
  const canRun = !busy && !!q.trim() && !waitingForClassifier;
  const primaryLabel = tool === "rewrite" ? `Rewrite to ${targetLevel}` : activeTool.label;
  const runActiveTool = () => {
    if (canRun) void run(activeTool.action);
  };
  const onQuestionKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runActiveTool();
    }
  };

  const level = mod?.bloom_level || bloom?.level;
  const hasResults = !!(level || mod || linguisticMod || targetRewrite?.rewritten || answer || summary);
  const loadingHint =
    busy === "Classifying the question" || busy === "Preparing moderation guidance"
      ? "Reviewing the question against local Bloom guidance."
      : "Your local model is preparing a grounded response.";
  const showSuccess = success.show && !busy;
  const classifierStatus = classifierLoaded
    ? { cls: "ready", text: "Bloom classifier ready" }
    : classifierFailed
      ? { cls: "error", text: "Bloom classifier unavailable" }
      : { cls: "", text: "Loading Bloom classifier…" };

  return (
    <main className="page-wrap fade-in teacher-workspace">
      <VideoHero
        eyebrow="Protected moderation studio"
        title={<>Question moderation,<br /><span>with academic intent intact.</span></>}
        description="Check the Bloom level of a question, rewrite it to another level, improve its wording, summarize text or ask the local model. Everything runs on this computer."
        actions={
          <>
            <button type="button" className="btn-primary" onClick={() => scrollToId("question-workspace", "teacher-input")}>
              Start with a question →
            </button>
            <button type="button" className="btn-secondary" onClick={() => scrollToId("reference-material")}>
              Add reference material
            </button>
          </>
        }
        trust={["01 Enter a question", "02 Pick a tool", "03 Review & confirm"]}
        clips={TEACHER_CLIPS}
      />

      <details id="reference-material" className="optional-panel glass-card mb-5 rounded-[26px] p-5 sm:p-7" open>
        <summary>
          <div>
            <h2 className="section-title">
              Reference material <span className="optional-tag">Optional</span>
            </h2>
            <p className="section-copy">
              Add course documents first if you want “Summarize text” and “Ask questions” to use them. Not needed for the Bloom tools, so you can collapse this and go straight to your question.
            </p>
          </div>
          <svg className="chevron" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </summary>
        <div className="mt-6 grid gap-5 md:grid-cols-2">
          <div className="grid gap-3 content-start">
            <p className="step-label"><b>A</b>Upload a file</p>
            <FilePicker file={file} onChange={setFile} disabled={!!busy} hint="PDF, TXT, MD, CSV, JPG or PNG" />
            <button type="button" className="btn-secondary" disabled={!!busy || !file} onClick={upload}>
              Add file
            </button>
          </div>
          <div className="grid gap-3 content-start">
            <label htmlFor="teacher-material" className="step-label"><b>B</b>Or paste text</label>
            <textarea
              id="teacher-material"
              className="min-h-32"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste lecture notes or a textbook section…"
            />
            <button type="button" className="btn-secondary" disabled={!!busy || !text.trim()} onClick={index}>
              Add pasted text
            </button>
          </div>
        </div>
        {materials.length > 0 && (
          <div className="mt-5">
            <p className="step-label">Added in this session</p>
            <ul className="material-list">
              {materials.map((m) => <li key={m.id}>{m.name}</li>)}
            </ul>
          </div>
        )}
        <button type="button" className="btn-quiet mt-4" disabled={!!busy} onClick={clearNotes}>
          Remove all material
        </button>
      </details>

      <section id="question-workspace" className="teacher-console guided glass-card rounded-[26px] p-5 sm:p-7" aria-labelledby="tool-heading">
        <div className="field-row">
          <div>
            <p className="eyebrow">Question workspace</p>
            <h2 id="tool-heading" className="section-title mt-3">What would you like to do?</h2>
          </div>
          <span className={`inline-status ${classifierStatus.cls}`} role="status">
            <i aria-hidden="true" />
            {classifierStatus.text}
          </span>
        </div>

        <div>
          <p className="step-label"><b>1</b>Choose a tool</p>
          <div className="ui-tabs" role="tablist" aria-label="Teacher tools">
            {TOOLS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                id={`tab-${t.id}`}
                aria-selected={tool === t.id}
                aria-controls="tool-panel"
                className="ui-tab"
                onClick={() => setTool(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div id="tool-panel" role="tabpanel" aria-labelledby={`tab-${tool}`} className="tool-panel">
          <p className="tool-intro">{activeTool.intro}</p>

          <div>
            <label htmlFor="teacher-input" className="step-label"><b>2</b>{activeTool.inputLabel}</label>
            <textarea
              id="teacher-input"
              className={tool === "summary" ? "min-h-48" : "min-h-32"}
              value={q}
              onChange={(e) => handleQuestionChange(e.target.value)}
              onKeyDown={onQuestionKeyDown}
              placeholder={activeTool.placeholder}
              aria-describedby="teacher-input-hint"
            />
            <span id="teacher-input-hint" className="field-hint">{activeTool.hint}</span>
          </div>

          {tool === "rewrite" && (
            <fieldset>
              <legend className="step-label"><b>3</b>Target Bloom level</legend>
              <div className="level-grid">
                {BLOOM_LEVELS.map((l) => (
                  <button
                    key={l.level}
                    type="button"
                    className="level-chip"
                    aria-pressed={targetLevel === l.level}
                    onClick={() => handleTargetLevelSelect(l.level)}
                    disabled={!!busy}
                  >
                    <strong>{l.level} <small className="text-[#66819e]">{l.code}</small></strong>
                    <span>{l.hint}</span>
                  </button>
                ))}
              </div>
            </fieldset>
          )}

          <div className="field-row">
            <span className="kbd-hint">
              {waitingForClassifier
                ? "This tool becomes available when the Bloom classifier has loaded."
                : !q.trim()
                  ? `Enter ${tool === "summary" ? "some text" : "a question"} to continue.`
                  : <>Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to run</>}
            </span>
            <button type="button" className="btn-primary" disabled={!canRun} onClick={runActiveTool}>
              {primaryLabel} →
            </button>
          </div>
        </div>
      </section>

      <section ref={resultsRef} className="mt-5" aria-live="polite" aria-label="Results">
        {!hasResults ? (
          <div className="results-empty">
            Results appear here. Nothing is sent outside this computer.
          </div>
        ) : (
          <div className="teacher-results">
            {level && (
              <article className="glass-card teacher-result-card rounded-[22px] p-5">
                <p className="eyebrow">Bloom classification</p>
                <div className="teacher-level">
                  <strong>{level}</strong>
                  <span>
                    {bloom
                      ? `${Math.round(bloom.confidence * 100)}% confidence`
                      : "Classified locally"}
                  </span>
                </div>
                <p className="section-copy">
                  Is this right? Confirm it, or choose the correct level. Confirmed levels stay on this
                  computer and help improve the classifier.
                </p>
                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <label htmlFor="confirm-level" className="sr-only">
                    Correct Bloom level
                  </label>
                  <select
                    id="confirm-level"
                    style={{ width: "auto" }}
                    value={confirmLevel || level}
                    onChange={(e) => {
                      setConfirmLevel(e.target.value);
                      setLabelNote("");
                    }}
                  >
                    {BLOOM_LEVELS.map((l) => (
                      <option key={l.level} value={l.level}>
                        {l.level} ({l.code})
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="btn-secondary"
                    disabled={labelSaving}
                    onClick={() => saveConfirmedLevel(level)}
                  >
                    {labelSaving ? "Saving…" : (confirmLevel && confirmLevel !== level) ? "Save correction" : "Confirm level"}
                  </button>
                </div>
                {labelNote && <p className="field-hint" role="status">{labelNote}</p>}
              </article>
            )}
            {mod && (
              <article className="glass-card teacher-result-card teacher-rewrite-card rounded-[22px] p-5">
                <p className="eyebrow">Higher level rewrite suggestion</p>
                <h2 className="section-title mt-2">Elevate to {mod.target_higher_level}</h2>
                <p className="teacher-reason">{mod.reason}</p>
                <div className="teacher-rewrite">
                  <span>Suggested rewrite</span>
                  <p>{mod.higher_level_rewrite}</p>
                </div>
              </article>
            )}
            {linguisticMod && (
              <article className="glass-card teacher-result-card rounded-[22px] p-5">
                <p className="eyebrow">Improved wording</p>
                <div className="teacher-comparison">
                  <div>
                    <span className="text-sm font-semibold text-[#66819e]">Original</span>
                    <p className="mt-2 text-sm leading-relaxed">{linguisticMod.original}</p>
                  </div>
                  <div className="text-center text-2xl text-[#416ce4]" aria-hidden="true">→</div>
                  <div>
                    <span className="text-sm font-semibold text-[#29795f]">Improved</span>
                    <p className="mt-2 text-sm leading-relaxed">{linguisticMod.improved}</p>
                  </div>
                </div>
                <button type="button" className="btn-quiet mt-3" onClick={() => handleQuestionChange(linguisticMod.improved)}>
                  Use improved version
                </button>
              </article>
            )}
            {targetRewrite?.rewritten && (
              <article className="glass-card teacher-result-card teacher-rewrite-card rounded-[22px] p-5">
                <p className="eyebrow">Target level rewrite</p>
                <h2 className="section-title mt-2">Rewritten for {targetRewrite.target_level}</h2>
                <div className="teacher-rewrite">
                  <span>Original</span>
                  <p className="text-sm text-gray-600">{targetRewrite.original}</p>
                </div>
                <div className="teacher-rewrite">
                  <span>Rewritten</span>
                  <p>{targetRewrite.rewritten}</p>
                </div>
                {targetRewrite.verified_level ? (
                  targetRewrite.verified_level.toLowerCase() === targetRewrite.target_level.toLowerCase() ? (
                    <p className="field-hint">
                      Checked by the Bloom classifier: reads as {targetRewrite.verified_level}
                      {targetRewrite.attempts && targetRewrite.attempts > 1 ? ` (best of ${targetRewrite.attempts} drafts)` : ""}. Still review it before use.
                    </p>
                  ) : (
                    <p className="field-hint" style={{ color: "var(--warn)" }}>
                      The Bloom classifier reads this rewrite as {targetRewrite.verified_level}, not {targetRewrite.target_level}
                      {targetRewrite.attempts && targetRewrite.attempts > 1 ? ` (closest of ${targetRewrite.attempts} drafts)` : ""}. Edit it or try again.
                    </p>
                  )
                ) : (
                  <p className="field-hint">Check the rewrite before use. To verify its level, paste it into “Check Bloom level”.</p>
                )}
              </article>
            )}
            {summary && (
              <article className="glass-card teacher-result-card rounded-[22px] p-5">
                <p className="eyebrow">Local summary</p>
                <p className="teacher-answer">{summary}</p>
              </article>
            )}
            {answer && (
              <article className="glass-card teacher-result-card rounded-[22px] p-5">
                <p className="eyebrow">Answer from the local model</p>
                <p className="teacher-answer">{answer}</p>
              </article>
            )}
          </div>
        )}
      </section>

      {error && (
        <ErrorModal
          title={errorTitle(error)}
          message={error}
          onClose={() => setError("")}
          action={
            !classifierLoaded && !busy ? (
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setError("");
                  setClassifierLoadAttempt((attempt) => attempt + 1);
                }}
              >
                Retry loading
              </button>
            ) : undefined
          }
        />
      )}


      {busy && <LoadingComponent label={busy} hint={loadingHint} showGame={true} onCancel={cancelOperation} progress={progress} />}
      {showSuccess && <LoadingComponent label="" hint="Operation completed successfully" success={true} successMessage={success.message} finalElapsed={success.elapsed} />}
    </main>
  );
}

export default TeacherStudio;
