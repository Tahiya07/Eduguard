from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH)

# ---------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------
hiddenimports = []

for package in [
    "backend",
    "backend.routes",
    "backend.service",
    "fastapi",
    "uvicorn",
    "pydantic",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "faiss",
    "llama_cpp",
]:
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

# ---------------------------------------------------------
# Explicit runtime data
# ---------------------------------------------------------
datas = []
binaries = []

# llama.cpp native libraries
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    binaries += collect_dynamic_libs("llama_cpp")
except Exception:
    pass

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------
a = Analysis(
    ["app_launcher.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unused optional packages/features
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

        # Optional ML ecosystems not used by EduGuard
        "tensorflow",
        "tensorflow_hub",
        "keras",
        "jax",
        "jaxlib",
        "flax",
        "mxnet",

        # Development / notebook integrations
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
    name="EduGuard",
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
    name="EduGuard",
)
