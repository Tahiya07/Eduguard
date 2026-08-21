import os
import sys
import time
import shutil
import socket
import threading
import subprocess
import webbrowser
from pathlib import Path



def get_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = get_root()

FRONTEND_DIR = ROOT / "frontend" / ".next" / "standalone"
BACKEND_EXE = ROOT / "EduGuardBackend.exe"

bundled_node = ROOT / "node.exe"
system_node = shutil.which("node")

if bundled_node.exists():
    NODE = bundled_node
elif system_node:
    NODE = Path(system_node)
else:
    NODE = None

frontend_process = None
backend_process = None


def build_environment():
    env = os.environ.copy()

    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["OFFLINE_MODE"] = "true"

    env["GENERATOR_MODEL_PATH"] = str(
        ROOT / "models" / "qwen.gguf"
    )

    env["BLOOM_MODEL_DIR"] = str(
        ROOT / "models" / "qwen_bloom_merged0.5B"
    )

    # IMPORTANT:
    # This is a profile name, not the filesystem path.
    env["RETRIEVAL_ENCODER"] = "bge-small"

    env["CORS_ORIGINS"] = (
        "http://127.0.0.1:3000,"
        "http://localhost:3000"
    )

    env["NEXT_PUBLIC_API_URL"] = "http://127.0.0.1:8000"

    return env


def port_ready(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def run_backend():
    global backend_process

    if getattr(sys, "frozen", False):
        backend_process = subprocess.Popen(
            [str(BACKEND_EXE)],
            cwd=str(ROOT),
            env=os.environ.copy(),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        python_exe = Path(sys.executable)

        backend_process = subprocess.Popen(
            [
                str(python_exe),
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=str(ROOT),
            env=os.environ.copy(),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )


def wait_for_backend(timeout=300):
    start = time.time()

    while time.time() - start < timeout:
        if port_ready("127.0.0.1", 8000):
            return True

        time.sleep(0.5)

    return False


def wait_for_frontend(timeout=60):
    import urllib.request

    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:3000/",
                timeout=1.5
            ) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            pass

        if frontend_process is not None:
            if frontend_process.poll() is not None:
                return False

        time.sleep(0.25)

    return False


def start():
    global frontend_process

    os.environ.update(build_environment())

    print("========================================")
    print("             EduGuard")
    print("========================================")
    print()

    if not NODE or not NODE.exists():
        raise RuntimeError("Node.js runtime could not be located.")

    server_file = FRONTEND_DIR / "server.js"

    if not server_file.exists():
        raise RuntimeError(
            f"Next.js server missing: {server_file}"
        )

    # ---------------------------------------------------------
    # START FRONTEND FIRST
    # ---------------------------------------------------------
    print("Starting frontend...")

    frontend_env = os.environ.copy()
    frontend_env["HOSTNAME"] = "127.0.0.1"
    frontend_env["PORT"] = "3000"

    frontend_process = subprocess.Popen(
        [
            str(NODE),
            "server.js",
        ],
        cwd=str(FRONTEND_DIR),
        env=frontend_env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    print("Waiting for frontend...")

    if not wait_for_frontend(timeout=60):
        raise RuntimeError("EduGuard frontend failed to start.")

    webbrowser.open("http://127.0.0.1:3000")
    print("Frontend ready.")
    print()
    print("Starting backend in background...")

    # ---------------------------------------------------------
    # START BACKEND IN BACKGROUND
    # ---------------------------------------------------------
    backend_thread = threading.Thread(
        target=run_backend,
        daemon=True,
    )

    backend_thread.start()

    print()
    print("========================================")
    print("        EduGuard is running")
    print("========================================")
    print()
    print("Frontend: http://127.0.0.1:3000")
    print("AI engine is initializing in background...")
    print()

    # ---------------------------------------------------------
    # MONITOR BACKEND WITHOUT BLOCKING THE UI
    # ---------------------------------------------------------
    backend_ready_reported = False
    backend_start_time = time.time()

    try:
        while True:

            if frontend_process.poll() is not None:
                raise RuntimeError(
                    "EduGuard frontend stopped unexpectedly."
                )

            if (
                not backend_ready_reported
                and port_ready("127.0.0.1", 8000)
            ):
                backend_ready_reported = True

                elapsed = time.time() - backend_start_time

                print(
                    f"AI engine ready ({elapsed:.1f}s)."
                )

            # Give backend a generous initialization period.
            if (
                not backend_ready_reported
                and time.time() - backend_start_time > 300
            ):
                print(
                    "WARNING: AI engine is taking longer than expected."
                )

            time.sleep(1)

    except KeyboardInterrupt:
        pass

def shutdown():
    global frontend_process, backend_process

    print()
    print("Stopping EduGuard...")

    if frontend_process is not None:
        if frontend_process.poll() is None:
            try:
                frontend_process.terminate()
                frontend_process.wait(timeout=5)
            except Exception:
                try:
                    frontend_process.kill()
                except Exception:
                    pass

    if backend_server is not None:
        try:
            backend_server.should_exit = True
        except Exception:
            pass


if __name__ == "__main__":
    try:
        start()
    except Exception as exc:
        print()
        print("EduGuard failed to start:")
        print(exc)
        input("\nPress Enter to exit...")
    finally:
        shutdown()













