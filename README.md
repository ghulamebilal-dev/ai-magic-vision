# AI Magic Vision - Disappearing Person Webcam Effect

A real-time, AI-powered "magic disappearance" webcam experience. Point your
laptop camera at yourself, raise an **open palm** and watch your body dissolve
into glowing particles, leaving only the clean background behind. Make a
**closed fist** and you materialize back out of the particles. A **pointing
finger** triggers a teleport glitch.

No buttons trigger the magic - your hand gestures do. Everything runs locally
on your laptop CPU; no video is ever uploaded anywhere.

## How it works

### 1. Background capture
When the app starts you are asked to step out of frame for 2 seconds. During
that window ~45 frames are collected and a per-pixel **median** frame becomes
the stable clean background (see `backend/background.py`). The same frames
also train the fallback background subtractor.

### 2. Hand gesture detection (`backend/gesture_detector.py`)
MediaPipe Hands finds 21 hand landmarks on a downscaled frame. A pure,
testable rule classifies the pose from landmark geometry - a finger counts as
*extended* when its tip is farther from the wrist than its PIP joint:

- **Open palm** (all fingers + thumb extended) -> disappear
- **Fist** (no fingers extended) -> reappear
- **Point** (index only, held 1 second) -> teleport glitch
- **Ghost mode** (1-4 fingers): the finger count drives your transparency
  *live* - 1 finger slightly faded, 2 more, 3 more, 4 almost gone.
  Lower your fingers (or make a fist) to come back to solid.

A gesture must be **held ~0.35 s** before it fires, so passing hand movements
never trigger the effect by accident.

### 3. Person segmentation (`backend/person_segmentation.py`)
Primary: **MediaPipe SelfieSegmentation** on a 256-px-wide frame. The raw
mask is cleaned with morphological open/close, feathered with a Gaussian
blur, and temporally smoothed with an exponential moving average - this kills
jagged edges, flicker and holes. If MediaPipe is unavailable, a **fallback**
path uses MOG2 background subtraction plus largest-contour filtering, so the
app still runs. The effect layer only ever sees a float mask, so the
segmentation method can be replaced without touching the effects.

### 4. The disappearance effect (`backend/effects.py`)
A state machine (`NORMAL -> DISAPPEARING -> HIDDEN -> REAPPEARING -> NORMAL`,
plus a `GLITCH` branch) drives a ~1.0 s cinematic sequence:

1. A golden glow ignites around your silhouette.
2. Your body turns slightly translucent.
3. A dissolve wave sweeps **bottom to top** with an organic noisy edge,
   spawning glowing gold/cyan particles along the wave front that drift
   upward and fade.
4. Your pixels are composited out using the clean background frame.
5. Only the original background remains (with a faint shimmer).

Reappearing reverses the idea: particles stream *inward* toward your
silhouette while a reveal wave sweeps top to bottom. The particle system is
pooled and capped (~450 particles) to stay CPU friendly.

**Complete disappearance (no shadow, no leftovers):** when you vanish, the
removal mask is the segmentation mask *plus* every pixel near you that differs
from the exposure-matched clean background. That catches the cast shadow,
hair, fingertips and any mask lag, so the whole area is replaced by the clean
scene. Once hidden, nothing is drawn on top - no particles, no shimmer. The
cached background is kept at full camera resolution and continuously
brightness-matched to the camera's auto-exposure, so there is no visible seam.

**Hand tracking glow (while hidden):** you are fully invisible, but your 21
hand landmarks are drawn as glowing dots + skeleton, one colour per finger
(thumb orange, index yellow, middle green, ring cyan, pinky magenta, palm
white). Turn it off with `HAND_TRACKING_FX = False` in `backend/effects.py`.

**AI overlays (while visible):** glowing irises that fire rotating light rays
(Eye Rays), a beating arc-reactor on the chest (Heart Reactor), and an
"AI SYSTEM ONLINE" boot sequence (scan line, flash, shock-wave) that plays
after the background is captured. Both effects have ON/OFF buttons in the UI
and the boot can be replayed. They fade out automatically when you vanish.
Tweak the heart position with `HEART_DX` / `HEART_DY` in `backend/overlays.py`.

**Ghost mode:** in the normal state, the number of extended fingers sets
your opacity live (1 finger = 0.75 visible, 2 = 0.55, 3 = 0.35, 4 = 0.18),
smoothed so it never jumps. Show a full open palm to trigger the complete
cinematic vanish instead.

### 5. Performance
- **Threaded pipeline:** a camera thread keeps only the newest frame, two worker
  threads run segmentation and hand tracking at their own pace, and one render
  thread composites + JPEG-encodes once for *all* browser tabs. Slow neural
  nets can no longer delay the picture.
- All mask math (clean-up, shadow/diff removal, wave) runs at ~1/3 size and is
  upsampled once; compositing is done in uint8 with OpenCV.
- Lite hand model (`model_complexity=0`), 256x144 segmentation input, MJPG
  camera mode + DirectShow on Windows, camera buffer size 1.
- Effects cost: ~25 ms -> ~3 ms per frame normally, ~120 ms -> ~5 ms while hidden.

## Project structure

```
ai-magic-vision/
├── backend/
│   ├── main.py               # FastAPI app: MJPEG stream + /api/* + serves frontend
│   ├── engine.py             # Orchestrator: camera -> bg -> mask -> gesture -> effect
│   ├── camera.py             # Webcam capture wrapper
│   ├── background.py         # Clean-background capture (median of ~45 frames)
│   ├── gesture_detector.py   # MediaPipe hands + rule-based gesture classification
│   ├── person_segmentation.py# MediaPipe selfie seg + fallback bg-subtraction
│   ├── effects.py            # Particle system + disappear/reappear/glitch engine
│   └── requirements.txt
├── frontend/
│   ├── index.html            # Dark glassmorphism UI
│   ├── style.css
│   └── app.js                # Status polling, HUD, buttons
├── tests/
│   └── test_pipeline.py      # Camera-free tests (synthetic frames)
├── run.py                    # One-command launcher
└── README.md
```

## Installation

**1. Create a virtual environment (Python 3.11+):**

Windows:
```bat
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

**3. Run:**
```bash
python run.py
```

The app opens automatically at http://127.0.0.1:8000. Grant camera access if
your OS asks.

## Using it

1. When the app starts, **step out of the camera frame for 2 seconds** while
   it captures the background ("Background captured. Step into the camera.").
2. Step back in. Raise an **open palm** toward the camera and hold it briefly
   - you dissolve into particles.
3. Make a **fist** and hold it briefly - you reappear.
4. Point with your **index finger** for a teleport glitch.
5. **Recalibrate Background**: step out of frame, click, wait 2 seconds.
6. **Restart Experience**: resets the effect and re-captures the background.

## Privacy

> Camera processing is local. No video is uploaded.

All processing happens in the Python process on your machine. Frames are
never sent to any server, and nothing is recorded or saved.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not open webcam" | Close other apps using the camera (Zoom/Teams), check the OS camera permission, then restart. |
| Gestures not detected | Hold your hand clearly in frame, palm facing the camera, with decent lighting. Detection runs on the mirrored (selfie) view. |
| Effect triggers accidentally | Gestures need a ~0.35 s hold; quick hand waves are ignored by design. |
| "Please make sure only one person is in the frame" | The app works best with one person; ask others to step out. |
| Low light warning | Move somewhere brighter; segmentation and hand tracking both degrade in the dark. |
| Background looks wrong after moving the laptop | Click **Recalibrate Background** and step out of frame for 2 s. |
| MediaPipe install fails | Use Python 3.11 and the pinned `numpy==1.26.4` from requirements. |
| Low FPS | Close heavy apps; the pipeline already downscales detection and strides detectors across frames. |

## Running the tests

The test suite needs no camera - it uses synthetic frames:

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

It covers gesture classification (open palm / fist / point), background
median capture, MediaPipe + fallback segmentation shapes, the full
disappear/reappear state machine, and the FastAPI routes.
