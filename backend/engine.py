"""VisionEngine: owns the camera, background model, detectors and effects.

Pipeline (three kinds of threads, so nothing waits for anything else):

    camera thread   : reads webcam frames, keeps only the NEWEST one
    worker threads  : person segmentation + hand gestures, each running as
                      fast as it can on the newest frame (results are
                      "latest wins", the render loop never waits for them)
    render thread   : newest frame + latest mask/gesture -> effect -> JPEG,
                      published once for ALL connected browsers

Before, everything ran one after another inside every HTTP stream, so a slow
neural-network call delayed the picture, and each extra stream (e.g. after
pressing a button) doubled the work.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .background import BackgroundModel
from .camera import WebcamCapture
from .effects import MagicEngine
from .face_tracker import FaceTracker
from .gesture_detector import NONE, GestureDetector
from .person_segmentation import PersonSegmenter


class _LatestWorker(threading.Thread):
    """Runs ``fn(frame)`` on the newest submitted frame, as fast as possible.

    Frames submitted while ``fn`` is busy are dropped (only the newest is
    kept), so a slow detector lowers its own update rate but never adds lag.
    """

    def __init__(self, fn, name: str):
        super().__init__(daemon=True, name=name)
        self._fn = fn
        self._cond = threading.Condition()
        self._frame = None
        self._stop_flag = False
        self._lock = threading.Lock()
        self.result = None
        self.count = 0

    def submit(self, frame: np.ndarray):
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def latest(self):
        with self._lock:
            return self.result

    def stop(self):
        with self._cond:
            self._stop_flag = True
            self._cond.notify()

    def run(self):
        while True:
            with self._cond:
                while self._frame is None and not self._stop_flag:
                    self._cond.wait()
                if self._stop_flag:
                    return
                frame, self._frame = self._frame, None
            try:
                res = self._fn(frame)
            except Exception:  # keep the worker alive on a bad frame
                continue
            with self._lock:
                self.result = res
                self.count += 1


class VisionEngine:
    JPEG_QUALITY = 75

    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 480,
                 disable_camera: bool = False):
        self.camera = WebcamCapture(cam_index, width, height)
        self.disable_camera = disable_camera
        self.camera_ok = False
        self.camera_error: str | None = None

        self.bg = BackgroundModel(num_frames=45, duration=2.0)
        self.segmenter = PersonSegmenter()
        self.gestures = GestureDetector()
        self.faces = FaceTracker()
        self._face = None
        self._face_worker: _LatestWorker | None = None
        self.fx: MagicEngine | None = None

        self.width = width
        self.height = height
        self._mask = np.zeros((height, width), dtype=np.float32)
        self._gesture = NONE
        self._finger_count = 0
        self._landmarks = None
        self._num_people = 0
        self._last_t = time.time()
        self.fps = 0.0
        self.message = ""
        self._low_light = False

        self._seg_worker: _LatestWorker | None = None
        self._hand_worker: _LatestWorker | None = None
        self._render_thread: threading.Thread | None = None
        self._running = False

        # Latest encoded frame, shared by every connected viewer.
        self._jpeg_cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """Open camera and begin background capture. Returns camera_ok."""
        if self.disable_camera:
            self.camera_error = "Camera disabled (test mode)."
            return False
        if not self.camera.open():
            self.camera_error = self.camera.error
            return False
        self.camera_ok = True
        self.height = self.camera.actual_height
        self.width = self.camera.actual_width
        self._mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.fx = MagicEngine(self.height, self.width)
        self.bg.start_capture()

        self._seg_worker = _LatestWorker(self.segmenter.segment, "segmenter")
        self._hand_worker = _LatestWorker(self.gestures.detect, "hands")
        self._face_worker = _LatestWorker(self.faces.track, "face")
        self._seg_worker.start()
        self._hand_worker.start()
        self._face_worker.start()

        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="render")
        self._render_thread.start()
        return True

    def shutdown(self):
        self._running = False
        for w in (self._seg_worker, self._hand_worker, self._face_worker):
            if w is not None:
                w.stop()
        with self._jpeg_cond:
            self._jpeg_cond.notify_all()
        self.camera.release()

    def recalibrate(self):
        """Ask the user to leave the frame; capture a fresh background."""
        if self.fx:
            self.fx.reset()
        self.segmenter.reset()
        self.bg.start_capture()

    def restart(self):
        self.recalibrate()

    # ---- optional AI overlays (toggled from the UI) ------------------- #
    FEATURES = ("eyes", "heart")

    def set_feature(self, name: str, on: bool) -> bool:
        if name not in self.FEATURES or self.fx is None:
            return False
        setattr(self.fx.overlays, f"{name}_on", bool(on))
        return True

    def replay_boot(self):
        if self.fx is not None:
            self.fx.overlays.start_boot()

    # ------------------------------------------------------------------ #
    # render loop + frame sharing
    # ------------------------------------------------------------------ #
    def _render_loop(self):
        while self._running:
            frame = self.process_frame()
            if frame is None:
                if self.camera.dead:
                    # Camera is gone: publish one error frame and idle.
                    self.camera_ok = False
                    self.camera_error = "Lost connection to the webcam."
                    self._publish(_error_frame(self.camera_error))
                    time.sleep(1.0)
                continue
            self._publish(frame)

    def _publish(self, frame: np.ndarray):
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        if not ok:
            return
        with self._jpeg_cond:
            self._jpeg = buf.tobytes()
            self._jpeg_seq += 1
            self._jpeg_cond.notify_all()

    def wait_jpeg(self, last_seq: int, timeout: float = 2.0):
        """Block until a frame newer than ``last_seq`` exists.

        Returns (seq, jpeg_bytes) or (last_seq, None) on timeout.
        """
        with self._jpeg_cond:
            if self._jpeg_seq == last_seq:
                self._jpeg_cond.wait(timeout)
            if self._jpeg_seq == last_seq or self._jpeg is None:
                return last_seq, None
            return self._jpeg_seq, self._jpeg

    # ------------------------------------------------------------------ #
    # per-frame processing
    # ------------------------------------------------------------------ #
    def process_frame(self) -> np.ndarray | None:
        """Wait for the next camera frame and return the processed output."""
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None
        return self._process(frame)

    def _process(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst

        # Hand the newest frame to the detectors (non-blocking) and pick up
        # whatever they finished most recently.
        if self._seg_worker is not None:
            self._seg_worker.submit(frame)
            self._hand_worker.submit(frame)
            if self.faces.ok and (self.fx.overlays.eyes_on
                                  or self.fx.overlays.heart_on):
                self._face_worker.submit(frame)
            self._face = self._face_worker.latest()
            seg = self._seg_worker.latest()
            if seg is not None:
                self._mask, self._num_people = seg
            hand = self._hand_worker.latest()
            if hand is not None:
                self._gesture, self._finger_count, self._landmarks = hand

        # --- background capture phase ---------------------------------- #
        if self.bg.capturing:
            # Guard: never finalize a background that contains a person.
            m = self._mask
            person_present = bool(m.shape[:2] == frame.shape[:2]
                                  and m.sum() > 0.02 * m.size)
            finished = self.bg.add_frame(frame, person_present)
            if not self.segmenter.mp_ok:
                self.segmenter.train_fallback(frame)
            if finished:
                self.segmenter.reset()
                self._mask = np.zeros_like(self._mask)
                self.fx.overlays.start_boot()   # "AI SYSTEM ONLINE"
            out = frame.copy()
            self._draw_capture_overlay(out, finished)
            return out

        background = self.bg.get(self.width, self.height)
        if background is None:  # should not happen, but stay alive
            return frame

        self._low_light = float(frame[::8, ::8].mean()) < 40.0
        self._update_message()

        dt = min(dt, 0.1)
        out, status = self.fx.update(frame, background, self._mask,
                                     self._gesture, self._finger_count, dt,
                                     landmarks=self._landmarks,
                                     face=self._face)
        self.fx.status = status
        return out

    # ------------------------------------------------------------------ #
    # status / overlays
    # ------------------------------------------------------------------ #
    def _update_message(self):
        if self._num_people > 1:
            self.message = "Please make sure only one person is in the frame."
        elif self._low_light:
            self.message = "Low light detected - move somewhere brighter."
        elif not self.segmenter.mp_ok:
            self.message = ("MediaPipe unavailable - using fallback "
                            "segmentation.")
        elif not self.gestures.ok:
            self.message = "Hand tracking unavailable - gestures disabled."
        else:
            self.message = ""

    def _draw_capture_overlay(self, frame: np.ndarray, finished: bool):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (5, 8, 18), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        if self.bg.warn_person:
            line1, line2 = "PERSON DETECTED IN FRAME!", "Please step OUT of the frame"
            color = (80, 80, 255)  # red in BGR
        elif finished:
            line1, line2 = "Background captured.", "Step into the camera."
            color = (255, 255, 255)
        else:
            secs = max(1, int(self.bg.remaining_seconds + 0.99))
            line1 = "Step OUT of the frame..."
            line2 = f"Capturing background: {secs}s"
            color = (255, 255, 255)
        for i, line in enumerate((line1, line2)):
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            x = (w - size[0]) // 2
            y = h // 2 - 20 + i * 48
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)
        # progress bar
        p = self.bg.progress
        bw = int(w * 0.6)
        x0 = (w - bw) // 2
        cv2.rectangle(frame, (x0, h // 2 + 60), (x0 + bw, h // 2 + 74),
                      (120, 120, 120), 1)
        cv2.rectangle(frame, (x0, h // 2 + 60),
                      (x0 + int(bw * p), h // 2 + 74), (0, 220, 255), -1)

    def snapshot_status(self) -> dict:
        if self.bg.capturing:
            effect = "Background Capturing..."
            message = ("Person detected - please step OUT of the frame!"
                       if self.bg.warn_person else "")
        elif not self.camera_ok:
            effect = "Camera Offline"
            message = self.message
        else:
            effect = self.fx.status if self.fx else "Starting..."
            message = self.message
        return {
            "camera_active": self.camera_ok,
            "fps": round(self.fps, 1),
            "gesture": self._gesture,
            "finger_count": self._finger_count,
            "effect_status": effect,
            "background_ready": self.bg.ready,
            "capturing": self.bg.capturing,
            "capture_progress": round(self.bg.progress, 3),
            "person_detected": bool(self._mask.sum() > 500),
            "num_people": self._num_people,
            "segmentation": ("mediapipe" if self.segmenter.mp_ok
                             else "fallback"),
            "message": message,
            "features": {
                "eyes": bool(self.fx and self.fx.overlays.eyes_on),
                "heart": bool(self.fx and self.fx.overlays.heart_on),
                "face_tracking": self.faces.ok,
            },
            "error": self.camera_error,
        }


def _error_frame(text: str):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Camera unavailable", (150, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 255), 2,
                cv2.LINE_AA)
    cv2.putText(img, text[:60], (60, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                cv2.LINE_AA)
    return img
