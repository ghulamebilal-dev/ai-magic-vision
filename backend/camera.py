"""Webcam capture with graceful failure handling.

Everything camera related lives here so the rest of the app never touches
cv2.VideoCapture directly. Frames are mirrored (selfie view) by default.

Speed: a background thread reads the camera continuously and keeps only the
*newest* frame. The processing loop therefore never works on stale buffered
frames (the classic cause of "laggy" webcam apps) and never blocks the camera.
"""
from __future__ import annotations

import sys
import threading

import cv2


class WebcamCapture:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 mirror: bool = True):
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap = None
        self.error: str | None = None
        self.actual_width = width
        self.actual_height = height

        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0          # increments for every new frame
        self._last_read = 0    # seq of the last frame handed out
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.dead = False      # reader gave up (camera unplugged / lost)

    # ------------------------------------------------------------------ #
    def _open_capture(self):
        """Try the fastest backend for the platform, then the default one."""
        backends = []
        if sys.platform.startswith("win"):
            backends.append(cv2.CAP_DSHOW)   # much faster start + FPS on Windows
        backends.append(cv2.CAP_ANY)
        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.index, backend)
            except Exception:  # pragma: no cover - hardware dependent
                continue
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return None

    def open(self) -> bool:
        """Open the webcam. Returns True on success, False with self.error set."""
        self.cap = self._open_capture()
        if self.cap is None:
            self.error = (
                "Could not open webcam. Make sure it is connected, not used by "
                "another app, and that you granted camera permission."
            )
            return False
        # MJPG gives full 30 FPS on most USB/laptop cameras (raw YUY2 is
        # often limited to 10-15 FPS at 640x480).
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warm up: discard the first few (often dark) frames.
        for _ in range(5):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.error = "Webcam opened but returned no frames."
            self.release()
            return False
        self.actual_height, self.actual_width = frame.shape[:2]

        self.dead = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name="camera-reader")
        self._thread.start()
        return True

    @property
    def is_open(self) -> bool:
        return (self.cap is not None and self.cap.isOpened()
                and not self.dead)

    def _reader(self):
        failures = 0
        while not self._stop.is_set():
            cap = self.cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 60:
                    self.dead = True
                    with self._cond:
                        self._cond.notify_all()
                    break
                self._stop.wait(0.02)
                continue
            failures = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        """Return (ok, frame): the next NEW frame (BGR, mirrored if set).

        Blocks until the camera delivers a frame newer than the previous one
        (so the caller is naturally paced to the camera FPS). Returns
        (False, None) on timeout or if the camera is lost.
        """
        if not self.is_open:
            return False, None
        with self._cond:
            if self._seq == self._last_read:
                self._cond.wait(timeout)
            if self._seq == self._last_read or self._frame is None:
                return False, None
            self._last_read = self._seq
            return True, self._frame

    def release(self):
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
