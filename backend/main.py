"""FastAPI app: serves the frontend, MJPEG video stream and control API.

All camera processing happens locally in this process. No frame is ever
uploaded anywhere.
"""
from __future__ import annotations

import atexit
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .engine import VisionEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

_engine: VisionEngine | None = None


def get_engine() -> VisionEngine:
    global _engine
    if _engine is None:
        disable = os.environ.get("MAGIC_NO_CAMERA") == "1"
        _engine = VisionEngine(disable_camera=disable)
        if not disable:
            _engine.start()
    return _engine


def _release():
    global _engine
    if _engine is not None:
        _engine.shutdown()


atexit.register(_release)

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app):
    import asyncio

    # Open camera / load models at startup, off the event loop.
    await asyncio.get_running_loop().run_in_executor(None, get_engine)
    yield


app = FastAPI(title="AI Magic Vision", lifespan=_lifespan)


def _mjpeg_generator():
    """Stream the engine's already-encoded frames (shared by all viewers)."""
    engine = get_engine()
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    seq = 0
    while True:
        seq, jpeg = engine.wait_jpeg(seq, timeout=2.0)
        if jpeg is None:
            if not engine._running:
                return
            continue
        yield boundary + jpeg + b"\r\n"


@app.get("/video_feed")
def video_feed():
    engine = get_engine()
    if not engine.camera_ok:
        raise HTTPException(
            status_code=503,
            detail=engine.camera_error or "Camera unavailable.",
        )
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def api_status():
    return get_engine().snapshot_status()


@app.post("/api/recalibrate")
def api_recalibrate():
    engine = get_engine()
    if not engine.camera_ok:
        raise HTTPException(status_code=503, detail="Camera unavailable.")
    engine.recalibrate()
    return {"ok": True, "message": "Step out of the frame - capturing..."}


@app.post("/api/restart")
def api_restart():
    engine = get_engine()
    if not engine.camera_ok:
        raise HTTPException(status_code=503, detail="Camera unavailable.")
    engine.restart()
    return {"ok": True, "message": "Experience restarted."}


@app.post("/api/feature/{name}")
def api_feature(name: str, on: bool = True):
    engine = get_engine()
    if not engine.camera_ok:
        raise HTTPException(status_code=503, detail="Camera unavailable.")
    if not engine.set_feature(name, on):
        raise HTTPException(status_code=404, detail="Unknown feature.")
    return {"ok": True, "message": f"{name} effect {'on' if on else 'off'}."}


@app.post("/api/boot")
def api_boot():
    engine = get_engine()
    if not engine.camera_ok:
        raise HTTPException(status_code=503, detail="Camera unavailable.")
    engine.replay_boot()
    return {"ok": True, "message": "Boot sequence replayed."}


# Serve the frontend last so /api/* routes take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
