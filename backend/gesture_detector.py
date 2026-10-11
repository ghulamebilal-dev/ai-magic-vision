"""Hand landmark detection + rule-based gesture classification.

Gestures:
  * open_palm -> disappear
  * fist      -> reappear
  * point     -> teleport/glitch effect (optional bonus)

Classification is a pure function of the 21 MediaPipe hand landmarks, so it
is unit-testable without a camera. Geometry used: a finger counts as
"extended" when its tip is farther from the wrist than its PIP joint.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# MediaPipe hand landmark indices.
_WRIST = 0
_THUMB_TIP, _THUMB_IP = 4, 3
_PINKY_MCP = 17
_FINGERS = {
    "index": (5, 6, 8),    # (mcp, pip, tip)
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

OPEN_PALM = "open_palm"
FIST = "fist"
POINT = "point"
NONE = "none"


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_states(landmarks):
    """Return (four_fingers_extended, thumb_extended) from 21 landmarks."""
    wrist = landmarks[_WRIST]

    def finger_extended(mcp_i, pip_i, tip_i) -> bool:
        tip, pip = landmarks[tip_i], landmarks[pip_i]
        return _dist(tip, wrist) > _dist(pip, wrist) * 1.05

    four = [
        finger_extended(mcp, pip, tip)
        for _, (mcp, pip, tip) in _FINGERS.items()
    ]
    thumb_tip, thumb_ip, pinky_mcp = (
        landmarks[_THUMB_TIP],
        landmarks[_THUMB_IP],
        landmarks[_PINKY_MCP],
    )
    thumb_out = _dist(thumb_tip, pinky_mcp) > _dist(thumb_ip, pinky_mcp) * 1.02
    return four, thumb_out


def count_extended_fingers(landmarks) -> int:
    """How many fingers are extended (0..5, includes thumb)."""
    if landmarks is None or len(landmarks) < 21:
        return 0
    four, thumb_out = _finger_states(landmarks)
    return int(sum(four) + (1 if thumb_out else 0))


def classify_hand(landmarks) -> str:
    """Classify from 21 (x, y) normalized landmarks. Returns a gesture name."""
    if landmarks is None or len(landmarks) < 21:
        return NONE
    four, thumb_out = _finger_states(landmarks)
    index, middle, ring, pinky = four
    if all(four) and thumb_out:
        return OPEN_PALM
    if not any(four):
        return FIST
    if index and not (middle or ring or pinky):
        return POINT
    return NONE


class GestureDetector:
    def __init__(self):
        self.ok = False
        self.init_error: str | None = None
        self._hands = None
        try:
            import mediapipe as mp  # local import: optional dependency

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,   # lite hand model: ~2x faster, accurate enough
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.ok = True
        except Exception as exc:
            self.init_error = str(exc)
            self.ok = False

    def detect(self, frame: np.ndarray):
        """Return (gesture, finger_count, landmarks) for the main hand.

        landmarks is a list of 21 normalized (x, y) points, or None.
        """
        if not self.ok:
            return NONE, 0, None
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (256, int(256 * h / w)),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return NONE, 0, None
        lm = result.multi_hand_landmarks[0].landmark
        points = [(p.x, p.y) for p in lm]
        return classify_hand(points), count_extended_fingers(points), points
