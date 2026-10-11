"""Face landmarks (iris centres, chin, forehead) via MediaPipe FaceMesh.

Used only for the decorative AI overlays (eye rays, heart reactor). It runs in
its own worker thread so it can never slow the video down. If MediaPipe is
missing the tracker simply reports ``ok = False`` and the overlays stay off.
"""
from __future__ import annotations

import cv2
import numpy as np


class FaceTracker:
    # FaceMesh landmark indices
    L_IRIS, R_IRIS = 468, 473          # iris centres (refine_landmarks=True)
    L_EYE = (33, 133)                  # eye corners (fallback if no iris)
    R_EYE = (263, 362)
    CHIN, FOREHEAD = 152, 10
    FACE_L, FACE_R = 234, 454

    def __init__(self):
        self.ok = False
        self.init_error: str | None = None
        self._mesh = None
        try:
            import mediapipe as mp

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.ok = True
        except Exception as exc:
            self.init_error = str(exc)

    def track(self, frame: np.ndarray):
        """Return a dict of normalized points for the main face, or None."""
        if not self.ok:
            return None
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, int(320 * h / w)),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        def pt(i):
            return (lm[i].x, lm[i].y)

        def eye(iris_i, corners):
            if len(lm) > iris_i:
                return pt(iris_i)
            a, b = pt(corners[0]), pt(corners[1])
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

        return {
            "eyes": [eye(self.L_IRIS, self.L_EYE), eye(self.R_IRIS, self.R_EYE)],
            "chin": pt(self.CHIN),
            "forehead": pt(self.FOREHEAD),
            "face_w": abs(lm[self.FACE_R].x - lm[self.FACE_L].x),
        }
