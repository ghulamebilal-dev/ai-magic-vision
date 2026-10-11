"""Camera-free tests for AI Magic Vision.

Everything here runs on synthetic numpy frames / landmarks, so the suite
works on machines with no webcam.
"""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.background import BackgroundModel
from backend.effects import MagicEngine
from backend.gesture_detector import classify_hand, count_extended_fingers
from backend.person_segmentation import PersonSegmenter


# --------------------------------------------------------------------- #
# synthetic hand landmarks (MediaPipe ordering)
# --------------------------------------------------------------------- #
def make_landmarks(pose):
    """21 (x, y) landmarks for 'palm', 'fist', 'point' or an int 0..4
    (that many fingers extended from the index side, thumb curled)."""
    lm = [(0.5, 0.5)] * 21
    lm = [list(p) for p in lm]
    lm[0] = [0.5, 0.9]  # wrist

    def finger(base_x, tip_y, pip_y, start):
        # mcp, pip, dip, tip
        lm[start] = [base_x, 0.70]
        lm[start + 1] = [base_x, pip_y]
        lm[start + 2] = [base_x, (pip_y + tip_y) / 2]
        lm[start + 3] = [base_x, tip_y]

    def thumb_curled():
        lm[1], lm[2], lm[3], lm[4] = (
            [0.44, 0.80], [0.46, 0.74], [0.48, 0.70], [0.50, 0.68])

    xs = [0.42, 0.50, 0.58, 0.66]  # index..pinky
    if isinstance(pose, int):
        for i, x in enumerate(xs):
            if i < pose:
                finger(x, 0.20, 0.50, 5 + i * 4)  # extended
            else:
                finger(x, 0.64, 0.50, 5 + i * 4)  # curled
        thumb_curled()
    elif pose == "palm":
        for i, x in enumerate(xs):
            finger(x, 0.20, 0.50, 5 + i * 4)  # tips far -> extended
        lm[1], lm[2], lm[3], lm[4] = [0.44, 0.80], [0.38, 0.70], [0.33, 0.60], [0.28, 0.50]
    elif pose == "fist":
        for i, x in enumerate(xs):
            finger(x, 0.64, 0.50, 5 + i * 4)  # tips near wrist -> curled
        thumb_curled()
    elif pose == "point":
        finger(xs[0], 0.20, 0.50, 5)  # index extended
        for i, x in enumerate(xs[1:], start=1):
            finger(x, 0.64, 0.50, 5 + i * 4)  # others curled
        thumb_curled()
    return [tuple(p) for p in lm]


def test_classify_open_palm():
    assert classify_hand(make_landmarks("palm")) == "open_palm"


def test_classify_fist():
    assert classify_hand(make_landmarks("fist")) == "fist"


def test_classify_point():
    assert classify_hand(make_landmarks("point")) == "point"


def test_classify_none_on_bad_input():
    assert classify_hand(None) == "none"
    assert classify_hand([]) == "none"


def test_count_extended_fingers():
    assert count_extended_fingers(make_landmarks("palm")) == 5
    assert count_extended_fingers(make_landmarks("fist")) == 0
    assert count_extended_fingers(make_landmarks("point")) == 1
    assert count_extended_fingers(make_landmarks(2)) == 2
    assert count_extended_fingers(make_landmarks(3)) == 3
    assert count_extended_fingers(make_landmarks(4)) == 4
    assert count_extended_fingers(None) == 0
    assert count_extended_fingers([]) == 0


# --------------------------------------------------------------------- #
# background model
# --------------------------------------------------------------------- #
def test_background_median_capture():
    bg = BackgroundModel(num_frames=8, duration=30.0)
    bg.start_capture()
    rng = np.random.default_rng(0)
    base = np.full((240, 320, 3), 100, dtype=np.uint8)
    done = False
    for _ in range(8):
        noisy = np.clip(base.astype(int) + rng.integers(-5, 6, base.shape),
                        0, 255).astype(np.uint8)
        done = bg.add_frame(cv2.resize(noisy, (640, 480)))
    assert done
    assert bg.ready
    got = bg.get(640, 480)
    assert got.shape == (480, 640, 3)
    assert abs(float(got.mean()) - 100) < 8


def test_background_rejects_contaminated_capture():
    """A background captured with a person in frame must be rejected."""
    bg = BackgroundModel(num_frames=8, duration=30.0)
    bg.start_capture()
    f = np.full((480, 640, 3), 100, dtype=np.uint8)
    # Person visible: frames are discarded, capture never finalizes.
    for _ in range(20):
        assert bg.add_frame(f, person_present=True) is False
    assert not bg.ready
    assert bg.warn_person
    # Person leaves: clean frames finalize the capture.
    done = False
    for _ in range(8):
        done = bg.add_frame(f, person_present=False)
    assert done and bg.ready
    assert not bg.warn_person


# --------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------- #
def test_segmentation_mediapipe_shape():
    seg = PersonSegmenter()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask, faces = seg.segment(frame)
    assert mask.shape == (480, 640)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert isinstance(faces, int)


def test_segmentation_fallback_detects_blob():
    seg = PersonSegmenter()
    seg.mp_ok = False  # force fallback path
    clean = np.full((480, 640, 3), 60, dtype=np.uint8)
    for _ in range(6):
        seg.train_fallback(clean)
    frame = clean.copy()
    cv2.rectangle(frame, (200, 100), (440, 440), (200, 200, 200), -1)
    mask, _ = seg.segment(frame)
    assert mask.sum() > 1000  # the rectangle must be detected as foreground


# --------------------------------------------------------------------- #
# effects state machine
# --------------------------------------------------------------------- #
def _scene(h=240, w=320):
    # Realistic scene: frame == background except for the "person" blob,
    # so exposure matching measures ~1.0 as it would in real use.
    bg = np.full((h, w, 3), 60, dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (w // 2, h // 2), 60, (140, 140, 140), -1)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), 60, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return frame, bg, mask


def test_disappear_reappear_state_machine():
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    dt = 1 / 30

    # hold open palm -> DISAPPEARING -> HIDDEN
    states = set()
    for _ in range(90):
        out, status = fx.update(frame, bg, mask, "open_palm", 5, dt)
        states.add(fx.state)
        assert out.shape == (240, 320, 3) and out.dtype == np.uint8
    assert "DISAPPEARING" in states
    assert fx.state == "HIDDEN", f"expected HIDDEN, got {fx.state}"

    # while hidden, person region must show the background
    out, _ = fx.update(frame, bg, mask, "none", 0, dt)
    inner = out[100:140, 140:180].astype(int)
    assert abs(inner.mean() - 60) < 25, "hidden person region should be background"

    # hold fist -> REAPPEARING -> NORMAL
    states = set()
    for _ in range(90):
        out, _ = fx.update(frame, bg, mask, "fist", 0, dt)
        states.add(fx.state)
    assert "REAPPEARING" in states
    assert fx.state == "NORMAL", f"expected NORMAL, got {fx.state}"


def test_weak_mask_still_fully_hidden():
    """Regression: even a weak/ghostly mask must fully hide the person."""
    fx = MagicEngine(240, 320)
    bg = np.full((240, 320, 3), 60, dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (160, 120), 60, (140, 140, 140), -1)
    # Simulate poor segmentation: soft low-confidence mask over the body.
    weak = np.zeros((240, 320), dtype=np.float32)
    cv2.circle(weak, (160, 120), 60, 0.45, -1)
    weak = cv2.GaussianBlur(weak, (15, 15), 0)
    dt = 1 / 30
    for _ in range(90):
        out, _ = fx.update(frame, bg, weak, "open_palm", 5, dt)
    assert fx.state == "HIDDEN", f"expected HIDDEN, got {fx.state}"
    out, _ = fx.update(frame, bg, weak, "none", 0, dt)
    inner = out[100:140, 140:180].astype(int)
    # Person region must now match the clean background almost exactly.
    assert abs(inner.mean() - 60) < 12, \
        f"ghost remains: region mean {inner.mean():.1f}, want ~60"


def test_glitch_effect_runs():
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    dt = 1 / 30
    seen = False
    for _ in range(60):
        out, _ = fx.update(frame, bg, mask, "point", 1, dt)
        assert out.shape == frame.shape
        if fx.state == "GLITCH":
            seen = True
    assert seen
    assert fx.state == "NORMAL"


def test_no_gesture_no_trigger():
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    for _ in range(60):
        out, _ = fx.update(frame, bg, mask, "none", 0, 1 / 30)
    assert fx.state == "NORMAL"


def test_ghost_mode_partial_transparency():
    """Finger count drives live transparency: 2 fingers = partial ghost."""
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()  # frame=140 gray, bg=60 gray
    dt = 1 / 30
    for _ in range(50):
        out, status = fx.update(frame, bg, mask, "none", 2, dt)
    assert fx.state == "NORMAL", "ghost mode must not trigger the vanish"
    assert "Ghost" in status
    inner = out[100:140, 140:180].astype(int).mean()
    # opacity target 0.55 -> 140*0.55 + 60*0.45 = 104
    assert 90 < inner < 120, f"ghost blend wrong: {inner:.1f}"
    # fist brings the person fully back
    for _ in range(50):
        out, _ = fx.update(frame, bg, mask, "fist", 0, dt)
    inner = out[100:140, 140:180].astype(int).mean()
    assert abs(inner - 140) < 8, f"did not return to solid: {inner:.1f}"


def test_shadow_suppression_in_hidden():
    """A cast shadow next to the person must vanish with the person."""
    fx = MagicEngine(240, 320)
    bg = np.full((240, 320, 3), 60, dtype=np.uint8)
    frame = np.full((240, 320, 3), 60, dtype=np.uint8)
    # person body (bright) and its cast shadow (dark patch beside it)
    cv2.circle(frame, (160, 120), 60, (140, 140, 140), -1)
    cv2.rectangle(frame, (225, 100), (250, 140), (20, 20, 20), -1)
    mask = np.zeros((240, 320), dtype=np.float32)
    cv2.circle(mask, (160, 120), 60, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    dt = 1 / 30
    for _ in range(90):
        out, _ = fx.update(frame, bg, mask, "open_palm", 5, dt)
    assert fx.state == "HIDDEN"
    out, _ = fx.update(frame, bg, mask, "none", 0, dt)
    body_region = out[100:140, 140:180].astype(int).mean()
    shadow_region = out[105:135, 230:248].astype(int).mean()
    assert abs(body_region - 60) < 12, f"body remains: {body_region:.1f}"
    assert abs(shadow_region - 60) < 14, \
        f"cast shadow remains visible: {shadow_region:.1f}"


def test_exposure_matching():
    """Cached background is brightness-matched to the live frame."""
    fx = MagicEngine(240, 320)
    bg_s = np.full((fx.lh, fx.lw, 3), 60, dtype=np.uint8)
    small = np.full((fx.lh, fx.lw, 3), 75, dtype=np.uint8)
    solid_s = np.zeros((fx.lh, fx.lw), dtype=np.float32)
    for _ in range(120):  # let the smoothed ratio converge
        fx._update_exposure(small, bg_s, solid_s)
    assert abs(60 * fx._exposure_ratio - 75) < 3, \
        f"exposure not matched: {60 * fx._exposure_ratio:.1f}"


def test_vanish_has_no_flashback():
    """The dissolve must go visible -> gone, never gone -> visible again."""
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    vals = []
    for _ in range(60):
        out, _ = fx.update(frame, bg, mask, "open_palm", 5, 1 / 30)
        if fx.state in ("DISAPPEARING", "HIDDEN"):
            vals.append(float(out[110:130, 150:170].mean()))
    # once the body region has reached the background it must stay there
    first_bg = next(i for i, v in enumerate(vals) if abs(v - 60) < 6)
    assert all(abs(v - 60) < 6 for v in vals[first_bg:]), vals


def test_hidden_screen_has_no_particles_or_residue():
    """HIDDEN shows only the clean scene: no particles, no shimmer."""
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    for _ in range(90):
        fx.update(frame, bg, mask, "open_palm", 5, 1 / 30)
    assert fx.state == "HIDDEN"
    for _ in range(30):
        out, _ = fx.update(frame, bg, mask, "none", 0, 1 / 30)
        assert fx.particles.n == 0
        assert np.abs(out.astype(int) - 60).max() < 8


def test_lagging_mask_still_fully_hidden():
    """Person moved but the (async) mask is stale: diff-removal must cover it."""
    fx = MagicEngine(240, 320)
    bg = np.full((240, 320, 3), 60, dtype=np.uint8)
    old = np.zeros((240, 320), np.float32)
    cv2.circle(old, (160, 120), 55, 1.0, -1)
    frame = bg.copy()
    cv2.circle(frame, (185, 125), 55, (150, 150, 150), -1)   # person moved
    for _ in range(90):
        fx.update(frame, bg, old, "open_palm", 5, 1 / 30)
    out, _ = fx.update(frame, bg, old, "none", 0, 1 / 30)
    assert np.abs(out.astype(int) - 60).max() < 12, "moved body still visible"


def test_hidden_shows_only_hand_tracking_points():
    """While hidden, coloured tracking dots appear at the hand - and only there."""
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    for _ in range(90):
        fx.update(frame, bg, mask, "open_palm", 5, 1 / 30)
    assert fx.state == "HIDDEN"
    lm = make_landmarks("palm")                      # normalized hand points
    for _ in range(20):
        out, _ = fx.update(frame, bg, mask, "none", 0, 1 / 30, landmarks=lm)
    diff = np.abs(out.astype(int) - 60).max(axis=2)
    ys, xs = np.nonzero(diff > 40)
    assert len(xs) > 30, "no tracking dots drawn"
    px = np.array(lm)[:, 0] * 320
    py = np.array(lm)[:, 1] * 240
    assert xs.min() >= px.min() - 60 and xs.max() <= px.max() + 60
    assert ys.min() >= py.min() - 60 and ys.max() <= py.max() + 60
    # hand lost -> dots fade away completely, scene is clean again
    for _ in range(30):
        out, _ = fx.update(frame, bg, mask, "none", 0, 1 / 30, landmarks=None)
    assert np.abs(out.astype(int) - 60).max() < 8


FACE = {"eyes": [(0.42, 0.30), (0.58, 0.30)], "chin": (0.50, 0.46),
        "forehead": (0.50, 0.18), "face_w": 0.20}


def test_overlays_draw_on_face_and_never_mutate_input():
    from backend.overlays import AIOverlays
    ov = AIOverlays(240, 320)
    frame = np.full((240, 320, 3), 60, np.uint8)
    keep = frame.copy()
    for _ in range(15):
        out = ov.draw(frame, FACE, 1 / 30, active=True)
    assert np.array_equal(frame, keep), "input frame was modified"
    diff = np.abs(out.astype(int) - 60).max(axis=2)
    ey, ex = int(0.30 * 240), int(0.42 * 320)
    assert diff[ey - 6:ey + 6, ex - 6:ex + 6].max() > 100, "eye glow missing"
    assert diff.max() > 100


def test_overlay_toggles_and_fade_when_inactive():
    from backend.overlays import AIOverlays
    ov = AIOverlays(240, 320)
    frame = np.full((240, 320, 3), 60, np.uint8)
    ov.eyes_on = ov.heart_on = False
    out = ov.draw(frame, FACE, 1 / 30, active=True)
    assert np.array_equal(out, frame), "disabled overlays still drew"
    ov.eyes_on = ov.heart_on = True
    for _ in range(10):
        ov.draw(frame, FACE, 1 / 30, active=True)
    for _ in range(30):                       # person vanishes -> not active
        out = ov.draw(frame, FACE, 1 / 30, active=False)
    assert np.array_equal(out, frame), "overlay left residue after fade"


def test_boot_sequence_runs_then_ends():
    from backend.overlays import AIOverlays
    ov = AIOverlays(240, 320)
    frame = np.full((240, 320, 3), 60, np.uint8)
    ov.start_boot()
    drew = False
    for _ in range(150):                      # 5 s at 30 fps
        out = ov.draw(frame, None, 1 / 30, active=True)
        drew |= not np.array_equal(out, frame)
    assert drew and ov.boot_t is None
    assert np.array_equal(ov.draw(frame, None, 1 / 30, True), frame)


def test_engine_hides_overlays_when_hidden_state():
    fx = MagicEngine(240, 320)
    frame, bg, mask = _scene()
    for _ in range(20):
        fx.update(frame, bg, mask, "none", 0, 1 / 30, face=FACE)
    for _ in range(100):
        fx.update(frame, bg, mask, "open_palm", 5, 1 / 30, face=FACE)
    assert fx.state == "HIDDEN"
    for _ in range(10):
        out, _ = fx.update(frame, bg, mask, "none", 0, 1 / 30, face=FACE)
    assert np.abs(out.astype(int) - 60).max() < 8


# --------------------------------------------------------------------- #
# FastAPI routes (camera disabled)
# --------------------------------------------------------------------- #
def test_api_routes():
    os.environ["MAGIC_NO_CAMERA"] = "1"
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("camera_active", "fps", "gesture", "finger_count", "effect_status",
                "background_ready", "person_detected", "message", "error"):
        assert key in body
    r = client.get("/")
    assert r.status_code == 200
    assert "AI MAGIC VISION" in r.text
