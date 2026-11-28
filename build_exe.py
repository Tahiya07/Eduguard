"""Build the portable EduGuard package.

Thin wrapper around build_exe.bat, which is the single source of truth for the build (frontend,
backend folder build, launcher, models, Tesseract, .env, verification). The previous Python build
path here was out of date: it copied the whole models\\ folder and used a different layout.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.exit(subprocess.call(["cmd", "/c", str(root / "build_exe.bat"), "--no-pause"], cwd=root))
