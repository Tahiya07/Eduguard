# PyInstaller spec for the EduGuard backend (FastAPI + llama.cpp + Bloom classifier + retrieval).
#
# Folder build ("onedir"): dist\EduGuardBackend\EduGuardBackend.exe + dist\EduGuardBackend\_internal\.
# build_exe.bat copies the whole folder to dist\EduGuard\backend\. A single-file build of
# torch/transformers would unpack several GB to %TEMP% on every start.
#
# Models are NOT bundled here; they live in dist\EduGuard\models\ and the launcher passes their paths.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)

hiddenimports = []
datas = []
binaries = []


def _no_tests(name: str) -> bool:
    # Test suites of numpy/scipy/sklearn/transformers etc. are never imported at runtime.
    parts = name.split(".")
    return not any(p in {"tests", "test", "testing", "_testing", "conftest"} for p in parts)

# Packages whose submodules are imported lazily (inside functions) or by name at runtime.
for package in [
    "backend",
    "privacy",
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic_settings",
    "multipart",          # python-multipart: file uploads
    "transformers",
    "tokenizers",
    "safetensors",
    "peft",               # Bloom classifier (LoRA-merged checkpoints still import peft)
    "accelerate",         # imported by peft at import time; excluding it breaks the classifier
    "faiss",
    "llama_cpp",
    "fitz",               # PyMuPDF: PDF uploads
    "pytesseract",        # image OCR (Tesseract itself is a separate program)
]:
    try:
        hiddenimports += collect_submodules(package, filter=_no_tests)
    except Exception:
        pass

# Top-level app modules imported lazily inside service functions.
hiddenimports += [
    "models", "rag_utils", "retriever", "encoder_backends", "retrieval_config", "ingestion",
    "predict_bloom", "bloom_prompt", "bloom_policy", "bloom_model_profiles",
    "role_access", "uncertainty", "operation_control", "runtime_utils", "summarizer",
    "multimodal_rag", "multi_slm", "qwen_gguf_cli",
    "hashing_vectorizer", "sectioned_summary", "summary_format", "input_validation",
]

for package in ["transformers", "tokenizers", "peft", "fitz"]:
    try:
        datas += collect_data_files(package, excludes=["**/tests/**", "**/test/**"])
    except Exception:
        pass

# The rewrite prompts are loaded by file path (bloom_prompt._load_v3_prompts), which PyInstaller
# cannot see; ship the file at the same relative location next to the bundled bloom_prompt module.
datas.append((str(ROOT / "experiments" / "multitask_bloom_rewrite" / "prompts.py"),
              "experiments/multitask_bloom_rewrite"))
# Word list for input_validation.py (rejects gibberish before classify/rewrite); read next to the module.
datas.append((str(ROOT / "input_vocab.txt"), "."))

for package in ["llama_cpp", "faiss", "fitz"]:
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

a = Analysis(
    ["backend_launcher.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "IPython", "jupyter", "notebook", "pytest", "pytest_asyncio",
        "tkinter", "wx", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "tensorflow", "tensorflow_hub", "keras", "jax", "jaxlib", "flax", "mxnet",
        "wandb", "tensorboard", "sphinx",
        # Training / research stacks are not needed at inference time.
        "torchaudio", "torchvision", "datasets", "evaluate", "trl", "bitsandbytes",
        "openai", "gradio", "easyocr",
        # Not imported by the app; pulled in only as optional extras of other packages (~190 MB).
        "nltk", "pyarrow", "pandas", "hf_xet", "IPython", "sympy.testing",
        # scikit-learn + SciPy (~100 MB): the retrieval fallback now uses hashing_vectorizer.py, an exact
        # copy of the HashingVectorizer it needed. transformers/torch only *check* whether they exist.
        "sklearn", "scipy",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EduGuardBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EduGuardBackend",
)
