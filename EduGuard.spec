# PyInstaller spec for EduGuard.exe, the small launcher that starts the frontend (node.exe) and the
# backend (backend\EduGuardBackend.exe) and opens the browser.
#
# Folder build: dist\EduGuard\EduGuard.exe + dist\EduGuard\_internal\. build_exe.bat then adds
# node.exe, backend\, frontend\, models\ and .env next to EduGuard.exe. node.exe is copied there by
# the build script (PyInstaller 6 would put a bundled binary inside _internal\, where the launcher
# does not look).
from pathlib import Path

ROOT = Path(SPECPATH)

a = Analysis(
    ["app_launcher.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "transformers", "numpy", "pandas", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
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
