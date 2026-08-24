"""
Simple script to build EduGuard EXEs with updated changes.
"""

import subprocess
import sys
from pathlib import Path

def build_exe(spec_file):
    """Build a single EXE from spec file."""
    print(f"Building {spec_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", spec_file],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {spec_file} built successfully")
        return True
    else:
        print(f"❌ {spec_file} build failed")
        print(result.stdout)
        print(result.stderr)
        return False

def copy_updated_files():
    """Copy updated files to dist folder."""
    import shutil
    
    root = Path(__file__).parent
    dist_root = root / "dist" / "EduGuard"
    
    print("Copying updated frontend build...")
    frontend_standalone = root / "frontend" / ".next" / "standalone"
    dist_frontend = dist_root / "frontend" / ".next"
    
    if frontend_standalone.exists():
        if dist_frontend.exists():
            shutil.rmtree(dist_frontend)
        shutil.copytree(frontend_standalone, dist_frontend)
        print("✅ Frontend build copied")
    
    print("Copying updated public folder...")
    frontend_public = root / "frontend" / "public"
    dist_public = dist_root / "frontend" / "public"
    
    if frontend_public.exists():
        if dist_public.exists():
            shutil.rmtree(dist_public)
        shutil.copytree(frontend_public, dist_public)
        print("✅ Public folder copied")
    
    print("Copying updated models...")
    models_dir = root / "models"
    dist_models = dist_root / "models"
    
    if models_dir.exists():
        if dist_models.exists():
            shutil.rmtree(dist_models)
        shutil.copytree(models_dir, dist_models)
        print("✅ Models copied")

def main():
    print("=" * 60)
    print("Building EduGuard EXEs with updated changes")
    print("=" * 60)
    print()
    
    # Build backend first
    if build_exe("EduGuardBackend.spec"):
        print()
        # Build main app
        if build_exe("EduGuard.spec"):
            print()
            # Copy updated files
            copy_updated_files()
            print()
            print("=" * 60)
            print("Build complete!")
            print("=" * 60)
            print()
            print("EXEs located in: dist/EduGuard/")
            print("- EduGuard.exe (main application)")
            print("- EduGuardBackend.exe (backend service)")
            print()
            print("Updated files included:")
            print("- Latest frontend build")
            print("- Latest models")
            print("- Latest backend code")
        else:
            print("Main app build failed")
    else:
        print("Backend build failed")

if __name__ == "__main__":
    main()
