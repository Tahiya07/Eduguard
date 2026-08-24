from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

hiddenimports = []
datas = []
binaries = []

packages = [
    "backend",
    "uvicorn",
    "fastapi",
    "pydantic",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "faiss",
    "llama_cpp",
    "multi_slm",
    "qwen_gguf_cli",
    "predict_bloom",
    "bloom_prompt",
]

for package in packages:
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

for package in [
    "sentence_transformers",
    "transformers",
    "tokenizers",
]:
    try:
        datas += collect_data_files(package)
    except Exception:
        pass

try:
    binaries += collect_dynamic_libs("llama_cpp")
except Exception:
    pass

# IMPORTANT:
# Models are intentionally NOT packaged into the backend EXE.
# They remain external under dist\EduGuard\models\.

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
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "pytest_asyncio",
        "tkinter",
        "wx",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "tensorflow",
        "tensorflow_hub",
        "keras",
        "jax",
        "jaxlib",
        "flax",
        "mxnet",
        "wandb",
        "tensorboard",
        "sphinx",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
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
