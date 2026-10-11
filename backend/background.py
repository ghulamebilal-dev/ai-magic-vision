"""Clean-background capture and storage.

On startup (and on recalibration) the user steps out of frame for ~2 seconds.
We collect a burst of frames and take the per-pixel median, which rejects
transient noise and gives a stable reference background used later to "erase"
the person.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


class BackgroundModel:
    def __init__(self, num_frames: int = 45, duration: float = 2.0):
        self.num_frames = num_frames
        self.duration = duration
        self.background: np.ndarray | None = None
        self._frames: list[np.ndarray] = []
        self._capturing = False
        self._start = 0.0
        # Set while a person is visible during capture: the capture is
        # contaminated (the "background" would contain the person!) so we
        # discard and wait for a clean frame.
        self.warn_person = False

    def start_capture(self):
        """Begin a fresh capture run. Previous background is kept until done."""
        self._frames = []
        self._capturing = True
        self._start = time.time()
        self.warn_person = False

    @property
    def capturing(self) -> bool:
        return self._capturing

    @property
    def ready(self) -> bool:
        return self.background is not None

    @property
    def progress(self) -> float:
        """0..1 progress of the current capture run (clean frames only)."""
        if not self._capturing:
            return 1.0 if self.ready else 0.0
        return float(min(1.0, len(self._frames) / max(1, self.num_frames)))

    @property
    def remaining_seconds(self) -> float:
        if not self._capturing:
            return 0.0
        return max(0.0, self.duration - (time.time() - self._start))

    def add_frame(self, frame: np.ndarray,
                  person_present: bool = False) -> bool:
        """Feed one BGR frame. Returns True when capture just finished.

        If a person is visible, the frame is rejected: the collected frames
        are discarded and the capture waits for a clean frame. This prevents
        the classic failure where the "clean background" contains the user,
        leaving a frozen body image behind after the vanish.
        """
        if not self._capturing:
            return False
        if person_present:
            self._frames = []
            self.warn_person = True
            self._start = time.time()
            return False
        self.warn_person = False
        # Keep full-resolution frames: a low-res background upscaled later
        # looks blurry next to the live video and leaves a visible "patch".
        self._frames.append(frame.copy())
        if len(self._frames) >= self.num_frames:
            self._finalize()
            return True
        # Safety backstop: never capture forever on a very slow camera.
        if time.time() - self._start > 10.0 and len(self._frames) >= 10:
            self._finalize()
            return True
        return False

    def _finalize(self):
        # A median of ~15 evenly spread frames is as clean as 45 and 3x faster.
        step = max(1, len(self._frames) // 15)
        used = self._frames[::step]
        stack = np.stack(used, axis=0)          # uint8, (n, h, w, 3)
        median = np.median(stack, axis=0)
        # Light denoise so the composited background looks clean.
        median = cv2.GaussianBlur(median.astype(np.float32), (3, 3), 0)
        self.background = np.clip(median, 0, 255).astype(np.uint8)
        self._frames = []
        self._capturing = False

    def get(self, width: int, height: int) -> np.ndarray | None:
        """Return the background resized to (width, height), or None."""
        if self.background is None:
            return None
        bg = self.background
        if bg.shape[1] != width or bg.shape[0] != height:
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)
            self.background = bg
        return bg
