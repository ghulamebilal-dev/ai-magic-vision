#!/usr/bin/env python3
"""One-command launcher for AI Magic Vision.

Checks the Python version and required packages, then starts the FastAPI
server and opens the app in your browser.

Usage:
    python run.py
"""
import os
import sys
import threading
import webbrowser

MIN_PY = (3, 11)
REQUIRED = ["cv2", "mediapipe", "numpy", "fastapi", "uvicorn"]

HOST = "127.0.0.1"
PORT = 8000


def main() -> int:
    if sys.version_info < MIN_PY:
        print(f"ERROR: Python 3.11+ is required (found {sys.version.split()[0]}).")
        return 1

    missing = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("ERROR: missing packages: " + ", ".join(missing))
        print("Install them with:")
        print("    pip install -r backend/requirements.txt")
        return 1

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    import uvicorn

    url = f"http://{HOST}:{PORT}"
    print("=" * 58)
    print("  AI MAGIC VISION - starting")
    print(f"  Open: {url}")
    print("  Press CTRL+C to stop.")
    print("=" * 58)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("backend.main:app", host=HOST, port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
