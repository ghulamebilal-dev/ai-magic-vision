"""Person segmentation layer.

Primary: MediaPipe SelfieSegmentation (fast on CPU, no GPU needed).
Fallback: background-subtraction + contour filtering, trained on the clean
background frames. The rest of the app only sees ``segment(frame)`` returning
a smooth float mask, so the method can be swapped without touching effects.

Speed: *all* mask clean-up (morphology, feathering, temporal smoothing,
contrast boost) happens at the model's tiny working size (256x144) and the
finished mask is upsampled to the camera size exactly once. Doing those steps
at 640x480 (as before) cost more than the neural network itself.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

# Working size of the segmentation model (landscape model = 256x144).
_SW, _SH = 256, 144


class PersonSegmenter:
    def __init__(self, temporal_alpha: float = 0.6):
        # Higher alpha = mask follows movement faster (less trailing "ghost").
        self.temporal_alpha = temporal_alpha
        self.smooth_mask: np.ndarray | None = None
        self.mp_ok = False
        self.init_error: str | None = None
        self._seg = None
        self._faces = None
        self._face_calls = 0
        self._last_face_count = 0
        self._lock = threading.Lock()

        try:
            import mediapipe as mp  # local import: optional dependency

            self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 1 = landscape/general model, good for full body
            )
            self._faces = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.55
            )
            self.mp_ok = True
        except Exception as exc:
            self.init_error = str(exc)
            self.mp_ok = False

        # Fallback: MOG2 background subtractor.
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=25, detectShadows=False
        )
        self._fallback_trained = False
        # Kernels are sized for the small working resolution.
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def train_fallback(self, frame: np.ndarray):
        """Feed a clean background frame to the fallback subtractor."""
        with self._lock:
            small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            self._mog.apply(small, learningRate=0.05)
            self._fallback_trained = True

    def segment(self, frame: np.ndarray):
        """Return (mask, num_people).

        mask: float32 array (h, w) in [0, 1], temporally smoothed, feathered.
        num_people: estimated face count (used for the multi-person warning).
        """
        h, w = frame.shape[:2]
        with self._lock:
            small = cv2.resize(frame, (_SW, _SH), interpolation=cv2.INTER_AREA)
            if self.mp_ok:
                raw = self._segment_mediapipe(small)
            else:
                raw = self._segment_fallback(frame)

            # Feather edges (small kernel: this is the 256x144 grid).
            feathered = cv2.GaussianBlur(raw, (5, 5), 0)

            # Temporal smoothing to kill flicker.
            a = self.temporal_alpha
            if (self.smooth_mask is None
                    or self.smooth_mask.shape != feathered.shape):
                self.smooth_mask = feathered
            else:
                self.smooth_mask = a * feathered + (1.0 - a) * self.smooth_mask
            mask = np.clip(self.smooth_mask, 0.0, 1.0)

            # Contrast boost: push weak-but-real detections toward solid 1 and
            # crush low noise toward 0.
            boosted = np.clip((mask - 0.15) / 0.35, 0.0, 1.0)
            boosted = boosted * boosted * (3.0 - 2.0 * boosted)

            num_people = self._count_faces(small)

        mask_full = cv2.resize(boosted.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)
        return mask_full, num_people

    def reset(self):
        with self._lock:
            self.smooth_mask = None

    # ------------------------------------------------------------------ #
    # MediaPipe path (runs on the 256x144 image)
    # ------------------------------------------------------------------ #
    def _segment_mediapipe(self, small_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._seg.process(rgb)
        m = result.segmentation_mask  # float32, higher = person
        if m is None:
            return np.zeros((_SH, _SW), dtype=np.float32)
        # Be inclusive on weak detections (soft ramp keeps sub-pixel edges);
        # the contrast boost later crushes the noise back to zero.
        soft = np.clip((m - 0.3) / 0.4, 0.0, 1.0).astype(np.float32)
        soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, self._k_open)
        soft = cv2.morphologyEx(soft, cv2.MORPH_CLOSE, self._k_close)
        return soft

    def _count_faces(self, small_bgr: np.ndarray) -> int:
        # Faces are cheap to detect; run every ~15 calls and cache.
        self._face_calls += 1
        if not self.mp_ok or self._face_calls % 15 != 1:
            return self._last_face_count
        try:
            rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self._faces.process(rgb)
            self._last_face_count = (
                len(result.detections) if result.detections else 0
            )
        except Exception:
            pass
        return self._last_face_count

    # ------------------------------------------------------------------ #
    # Fallback path: background subtraction + largest-contour filtering
    # ------------------------------------------------------------------ #
    def _segment_fallback(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        lr = 0 if self._fallback_trained else 0.05
        fg = self._mog.apply(small, learningRate=lr)
        _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_small = np.zeros((240, 320), dtype=np.float32)
        if contours:
            # Keep the largest contour if it is plausibly a person.
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) > 0.015 * (320 * 240):
                cv2.drawContours(mask_small, [biggest], -1, 1.0, cv2.FILLED)
        return cv2.resize(mask_small, (_SW, _SH),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
