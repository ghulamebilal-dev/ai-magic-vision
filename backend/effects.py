"""Cinematic disappear / reappear / glitch effects + particle system.

The effect layer only ever receives: the live frame, the clean background,
a float person mask, and the current gesture string. It knows nothing about
*how* the mask was produced, so segmentation can be swapped freely.

State machine:
    NORMAL -> (open palm held) -> DISAPPEARING -> HIDDEN
    HIDDEN -> (fist held)      -> REAPPEARING  -> NORMAL
    NORMAL -> (point held)     -> GLITCH       -> NORMAL

Disappearance (~1.0 s):
    1. golden glow ignites around the silhouette
    2. body turns slightly translucent
    3. a dissolve wave sweeps bottom -> top, spawning glowing particles
    4. everything the person left in the picture is replaced by the clean
       background - body, hair, hand AND cast shadow - and NOTHING is drawn
       on top of it afterwards.

How the person is *completely* removed
--------------------------------------
Segmentation alone is never perfect (lagging edges, hair, shadows). So the
removal mask is the union of
    (a) the segmentation mask, grown a few pixels, and
    (b) every pixel inside a generous zone around the person that differs
        from the exposure-matched clean background (this is what catches the
        cast shadow, soft hair edges and mask lag).
All of that is computed on a ~1/3-size image and upsampled once, which is
also what keeps the effect cheap enough for real-time use.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .overlays import AIOverlays


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# BGR particle palette: gold, cyan-white, warm white.
_PALETTE = np.array(
    [
        [0.25, 0.75, 1.0],   # gold
        [1.0, 0.95, 0.55],   # cyan-white
        [0.9, 0.9, 1.0],     # warm white
    ],
    dtype=np.float32,
)


class ParticleSystem:
    """Small pooled particle system (a few hundred particles, CPU friendly)."""

    def __init__(self, max_n: int = 450):
        self.max_n = max_n
        self.pos = np.zeros((max_n, 2), dtype=np.float32)
        self.vel = np.zeros((max_n, 2), dtype=np.float32)
        self.life = np.zeros(max_n, dtype=np.float32)
        self.max_life = np.ones(max_n, dtype=np.float32)
        self.size = np.zeros(max_n, dtype=np.float32)
        self.color = np.zeros((max_n, 3), dtype=np.float32)
        self.n = 0

    def clear(self):
        self.n = 0

    def spawn(self, pos, vel, life, size, color):
        k = int(min(len(pos), self.max_n - self.n))
        if k <= 0:
            return
        i0, i1 = self.n, self.n + k
        self.pos[i0:i1] = pos[:k]
        self.vel[i0:i1] = vel[:k]
        self.life[i0:i1] = life[:k]
        self.max_life[i0:i1] = np.maximum(life[:k], 1e-3)
        self.size[i0:i1] = size[:k]
        self.color[i0:i1] = color[:k]
        self.n = i1

    def update(self, dt: float, gravity: float = 0.0):
        if self.n == 0:
            return
        self.life[: self.n] -= dt
        alive = self.life[: self.n] > 0
        kept = int(np.count_nonzero(alive))
        if kept < self.n:
            # Compact arrays so dead particles vanish.
            self.pos[:kept] = self.pos[: self.n][alive]
            self.vel[:kept] = self.vel[: self.n][alive]
            self.life[:kept] = self.life[: self.n][alive]
            self.max_life[:kept] = self.max_life[: self.n][alive]
            self.size[:kept] = self.size[: self.n][alive]
            self.color[:kept] = self.color[: self.n][alive]
            self.n = kept
        if self.n:
            self.pos[: self.n] += self.vel[: self.n] * dt
            if gravity:
                self.vel[: self.n, 1] += gravity * dt

    def draw(self, frame: np.ndarray):
        """Additively draw all particles onto ``frame`` (uint8 BGR, in place)."""
        n = self.n
        if n == 0:
            return
        h, w = frame.shape[:2]
        ratios = (self.life[:n] / self.max_life[:n])
        xs = self.pos[:n, 0].astype(np.int32).tolist()
        ys = self.pos[:n, 1].astype(np.int32).tolist()
        radii = np.maximum(1, (self.size[:n] * ratios + 0.5)).astype(np.int32).tolist()
        cols = (self.color[:n] * (ratios[:, None] * 230.0)).astype(np.int32).tolist()
        overlay = np.zeros_like(frame)
        for x, y, r, c in zip(xs, ys, radii, cols):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (x, y), r, c, -1, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)


class MagicEngine:
    DISAPPEAR_DUR = 1.0
    REAPPEAR_DUR = 0.9
    GLITCH_DUR = 0.45
    HOLD_TIME = 0.35  # gesture must be held this long to trigger
    POINT_GLITCH_HOLD = 1.0  # point held this long -> teleport glitch
    # Ghost mode: how visible the person stays per extended-finger count.
    GHOST_OPACITY = {0: 1.0, 1: 0.75, 2: 0.55, 3: 0.35, 4: 0.18}

    BAND = 70.0            # softness (px) of the dissolve wave front
    NOISE_AMP = 110.0      # raggedness (px) of the wave front
    DIFF_THRESH = 12       # 0-255: "differs from clean background" level
    ROI_RADIUS = 60        # px around the person searched for shadow/hair
    # While hidden, show ONLY the tracked hand points (coloured, glowing).
    HAND_TRACKING_FX = True

    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width
        # All mask math runs at ~1/3 size.
        self.lw = max(32, width // 3)
        self.lh = max(24, int(round(height * self.lw / width)))
        self.sx = width / self.lw
        self.sy = height / self.lh

        self.state = "NORMAL"
        self.state_t = 0.0
        self.palm_hold = 0.0
        self.fist_hold = 0.0
        self.point_hold = 0.0
        self.ghost_opacity = 1.0  # 1 = fully visible, 0 = fully transparent
        self._exposure_ratio = 1.0  # smoothed bg brightness match factor
        self.particles = ParticleSystem()
        self.status = "Background Ready"
        self.overlays = AIOverlays(height, width)
        self._hand_pts: np.ndarray | None = None   # smoothed 21x2 (px)
        self._hand_vis = 0.0

        # Static organic noise so dissolve edges look natural, not straight.
        noise_small = np.random.rand(max(1, height // 8),
                                     max(1, width // 8)).astype(np.float32)
        self.noise_s = cv2.resize(noise_small, (self.lw, self.lh),
                                  interpolation=cv2.INTER_LINEAR)
        # Row coordinates (in full-res pixels) of the low-res rows.
        self.yy_s = ((np.arange(self.lh, dtype=np.float32) + 0.5)
                     * self.sy)[:, None]

        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        r = max(3, int(round(self.ROI_RADIUS / self.sx)))
        self._k_roi = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                (2 * r + 1, 2 * r + 1))
        self._rng = np.random.default_rng()

        # Cached low-res copy of the (static) clean background.
        self._bg_ref: np.ndarray | None = None
        self._bg_s: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #
    def reset(self):
        self.state = "NORMAL"
        self.state_t = 0.0
        self.palm_hold = self.fist_hold = self.point_hold = 0.0
        self.ghost_opacity = 1.0
        self._exposure_ratio = 1.0
        self._hand_pts = None
        self._hand_vis = 0.0
        self.overlays.reset()
        self.particles.clear()
        self.status = "Background Ready"

    # ------------------------------------------------------------------ #
    # main update
    # ------------------------------------------------------------------ #
    def update(self, frame: np.ndarray, background: np.ndarray,
               mask: np.ndarray, gesture: str, finger_count: int, dt: float,
               landmarks=None, face=None):
        """Process one frame. Returns (output_bgr, status_text)."""
        dt = min(max(dt, 1e-4), 0.1)
        self.state_t += dt

        # ---- cheap low-res products shared by every state ---------------- #
        small = cv2.resize(frame, (self.lw, self.lh),
                           interpolation=cv2.INTER_AREA)
        if self._bg_ref is not background:
            self._bg_ref = background
            self._bg_s = cv2.resize(background, (self.lw, self.lh),
                                    interpolation=cv2.INTER_AREA)
        if mask.shape != (self.lh, self.lw):
            mask_s = cv2.resize(mask.astype(np.float32), (self.lw, self.lh),
                                interpolation=cv2.INTER_AREA)
        else:
            mask_s = mask.astype(np.float32)
        solid_s, strong_s = self._solidify(mask_s)
        self._update_exposure(small, self._bg_s, strong_s)
        person_area = float((solid_s > 0.5).sum()) * self.sx * self.sy

        out = frame  # replaced below when something is composited

        if self.state == "NORMAL":
            self._update_holds(gesture, dt, allow=("open_palm", "point"))
            if self.palm_hold >= self.HOLD_TIME:
                self._begin("DISAPPEARING")
                self.status = "Disappearing..."
            elif self.point_hold >= self.POINT_GLITCH_HOLD:
                self._begin("GLITCH")
                self.status = "Teleport Glitch!"
            else:
                # Ghost mode: finger count drives transparency live.
                # 1 finger = slightly faded ... 4 fingers = almost gone.
                # Full open palm (5) triggers the cinematic vanish above.
                if gesture == "open_palm":
                    # "Charging" dip while the palm hold builds up.
                    target = 1.0 - 0.6 * (self.palm_hold / self.HOLD_TIME)
                else:
                    target = self.GHOST_OPACITY.get(finger_count, 1.0)
                k = min(1.0, dt * 8.0)
                self.ghost_opacity += (target - self.ghost_opacity) * k
                replace = 1.0 - self.ghost_opacity
                if replace > 0.02:
                    removal_s = self._removal(small, solid_s, strong_s)
                    out = self._composite(frame, background,
                                          removal_s * replace)
                if 1 <= finger_count <= 4:
                    plural = "s" if finger_count > 1 else ""
                    self.status = f"Ghost Mode ({finger_count} finger{plural})"
                elif self.palm_hold > 0.05:
                    self.status = "Open Palm Detected"
                elif self.point_hold > 0.05:
                    self.status = "Point Detected"
                else:
                    self.status = ("Person Detected" if person_area > 500
                                   else "Background Ready")

        elif self.state == "DISAPPEARING":
            removal_s = self._removal(small, solid_s, strong_s)
            out = self._render_disappearing(frame, background, removal_s,
                                            solid_s)
            self.status = "Disappearing..."
            if self.state_t >= self.DISAPPEAR_DUR:
                self._begin("HIDDEN")
                self.status = "Hidden"

        elif self.state == "HIDDEN":
            # Person + hair + hand + cast shadow -> clean background.
            # Nothing else is drawn: the screen shows only the scene.
            removal_s = self._removal(small, solid_s, strong_s)
            out = self._composite(frame, background, removal_s)
            if self.HAND_TRACKING_FX:
                out = self._draw_hand_tracking(out, landmarks, dt)
            self._update_holds(gesture, dt, allow=("fist",))
            if self.fist_hold >= self.HOLD_TIME:
                self._begin("REAPPEARING")
                self.status = "Reappearing..."
            elif self.fist_hold > 0.05:
                self.status = "Fist Detected"
            else:
                self.status = "Hidden"

        elif self.state == "REAPPEARING":
            removal_s = self._removal(small, solid_s, strong_s)
            out = self._render_reappearing(frame, background, removal_s,
                                           solid_s)
            self.status = "Reappearing..."
            if self.state_t >= self.REAPPEAR_DUR:
                self._begin("NORMAL")
                self.status = "Person Detected"

        elif self.state == "GLITCH":
            p = min(1.0, self.state_t / self.GLITCH_DUR)
            out = self._render_glitch(frame, p)
            self.status = "Teleport Glitch!"
            if self.state_t >= self.GLITCH_DUR:
                self._begin("NORMAL")

        # AI eye rays / heart reactor / boot sequence (fade out automatically
        # in every state except NORMAL, so a hidden person stays clean).
        out = self.overlays.draw(out, face, dt, active=(self.state == "NORMAL"))

        self.particles.update(dt)
        if self.particles.n:
            if out is frame:
                out = frame.copy()
            self.particles.draw(out)
        return out, self.status

    # ------------------------------------------------------------------ #
    # mask helpers (all low-res)
    # ------------------------------------------------------------------ #
    def _begin(self, state: str):
        self.state = state
        self.state_t = 0.0
        self.palm_hold = self.fist_hold = self.point_hold = 0.0
        if state == "NORMAL":
            self.ghost_opacity = 1.0
        if state == "HIDDEN":
            # Nothing may remain on screen once the person is gone.
            self.particles.clear()
        else:
            self._hand_pts = None
            self._hand_vis = 0.0

    def _solidify(self, feather_s: np.ndarray):
        """Soft mask -> (solid, strong) compositing masks (low-res).

        Both reach a full 1.0 inside the body so it is *completely* replaced
        (no ghost leftovers). ``strong`` is grown a bit more to also cover
        silhouette edges, hair and finger tips.
        """
        m = np.clip((feather_s - 0.08) / 0.30, 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)
        solid = cv2.dilate(m, self._k3)
        strong = cv2.dilate(m, self._k3, iterations=2)
        return solid.astype(np.float32), strong.astype(np.float32)

    def _update_holds(self, gesture: str, dt: float, allow):
        targets = {"open_palm": "palm_hold", "fist": "fist_hold",
                   "point": "point_hold"}
        cap = max(self.HOLD_TIME, self.POINT_GLITCH_HOLD) + 0.5
        for name, attr in targets.items():
            if name not in allow:
                setattr(self, attr, 0.0)
                continue
            val = getattr(self, attr)
            if gesture == name:
                val = min(val + dt, cap)
            else:
                val = max(0.0, val - dt * 2.0)
            setattr(self, attr, val)

    def _update_exposure(self, small: np.ndarray, bg_s: np.ndarray,
                         strong_s: np.ndarray):
        """Track the camera's auto-exposure so the cached background matches.

        Laptop webcams constantly re-expose when you step in/out of frame.
        Without compensation the replaced region has a brightness seam that
        looks like a lingering "shadow". The scale is the *median per-pixel
        ratio* over confident background pixels (robust to a cast shadow),
        clamped and temporally smoothed.
        """
        region = cv2.dilate(strong_s, self._k_roi) < 0.5
        if int(region.sum()) < 1500:
            region = strong_s < 0.3
            if int(region.sum()) < 300:
                return
        fr = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)[region].astype(np.float32)
        bg = cv2.cvtColor(bg_s, cv2.COLOR_BGR2GRAY)[region].astype(np.float32)
        valid = bg > 20.0
        if int(valid.sum()) < 200:
            return
        ratio = float(np.median(fr[valid] / bg[valid]))
        ratio = float(np.clip(ratio, 0.75, 1.33))
        self._exposure_ratio += (ratio - self._exposure_ratio) * 0.15

    def _removal(self, small: np.ndarray, solid_s: np.ndarray,
                 strong_s: np.ndarray) -> np.ndarray:
        """Low-res float mask of EVERYTHING the person leaves in the frame.

        = segmentation (grown)  U  anything near the person that differs from
        the exposure-matched clean background (cast shadow, hair, mask lag).
        """
        bg_m = cv2.convertScaleAbs(self._bg_s, alpha=self._exposure_ratio)
        roi = cv2.dilate(strong_s, self._k_roi) > 0.02
        diff = cv2.absdiff(small, bg_m)
        b, g, r = cv2.split(diff)
        d = cv2.max(cv2.max(b, g), r)
        changed = ((d > self.DIFF_THRESH) & roi).astype(np.uint8)
        # Drop 1-pixel sensor-noise speckle, then grow to cover soft edges.
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, self._k3)
        changed = cv2.dilate(changed, self._k3, iterations=2)
        removal = np.maximum(strong_s, changed.astype(np.float32))
        return cv2.GaussianBlur(removal, (5, 5), 0)

    # ------------------------------------------------------------------ #
    # full-res compositing
    # ------------------------------------------------------------------ #
    def _composite(self, frame: np.ndarray, background: np.ndarray,
                   alpha_s: np.ndarray) -> np.ndarray:
        """out = frame*(1-a) + exposure_matched_background*a   (uint8, fast)."""
        a = cv2.resize(np.clip(alpha_s, 0.0, 1.0), (self.w, self.h),
                       interpolation=cv2.INTER_LINEAR)
        a8 = cv2.convertScaleAbs(a, alpha=255.0)
        a3 = cv2.merge([a8, a8, a8])
        bg_m = cv2.convertScaleAbs(background, alpha=self._exposure_ratio)
        keep = cv2.multiply(frame, cv2.bitwise_not(a3), scale=1.0 / 255.0)
        fill = cv2.multiply(bg_m, a3, scale=1.0 / 255.0)
        return cv2.add(keep, fill)

    # BGR colour per finger: thumb orange, index yellow, middle green,
    # ring cyan, pinky magenta, palm white.
    _FINGER_COLORS = [(0, 150, 255), (0, 240, 255), (60, 255, 60),
                      (255, 230, 0), (255, 60, 255)]
    _PALM_COLOR = (255, 255, 255)
    _BONES = [
        (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
        (0, 5, 5), (5, 6, 1), (6, 7, 1), (7, 8, 1),
        (5, 9, 5), (9, 10, 2), (10, 11, 2), (11, 12, 2),
        (9, 13, 5), (13, 14, 3), (14, 15, 3), (15, 16, 3),
        (13, 17, 5), (17, 18, 4), (18, 19, 4), (19, 20, 4), (0, 17, 5),
    ]  # (from, to, colour index; 5 = palm)

    def _draw_hand_tracking(self, out: np.ndarray, landmarks, dt: float):
        """Coloured, glowing hand-tracking points + skeleton (only the hand)."""
        if landmarks is not None and len(landmarks) >= 21:
            pts = np.array(landmarks[:21], dtype=np.float32)
            pts[:, 0] *= self.w
            pts[:, 1] *= self.h
            if self._hand_pts is None:
                self._hand_pts = pts
            else:  # smooth the ~20 Hz detections so dots glide, not jitter
                self._hand_pts += (pts - self._hand_pts) * 0.6
            self._hand_vis = min(1.0, self._hand_vis + dt * 12.0)
        else:
            self._hand_vis = max(0.0, self._hand_vis - dt * 6.0)
        if self._hand_pts is None or self._hand_vis <= 0.02:
            self._hand_pts = None
            return out

        P = self._hand_pts
        vis = self._hand_vis
        x0 = int(max(0, P[:, 0].min() - 50)); x1 = int(min(self.w, P[:, 0].max() + 50))
        y0 = int(max(0, P[:, 1].min() - 50)); y1 = int(min(self.h, P[:, 1].max() + 50))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return out
        layer = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        off = np.array([x0, y0], dtype=np.float32)

        def col(ci):
            c = self._PALM_COLOR if ci == 5 else self._FINGER_COLORS[ci]
            return tuple(int(v * vis) for v in c)

        for a, b, ci in self._BONES:
            pa = tuple(int(v) for v in (P[a] - off))
            pb = tuple(int(v) for v in (P[b] - off))
            cv2.line(layer, pa, pb, col(ci), 2, cv2.LINE_AA)
        pulse = 1.0 + 0.25 * math.sin(self.state_t * 9.0)
        for i in range(21):
            ci = 5 if i == 0 else min(4, (i - 1) // 4)
            c = col(ci)
            tip = i in (4, 8, 12, 16, 20)
            pt = tuple(int(v) for v in (P[i] - off))
            r = int((7 if tip else 4) * (pulse if tip else 1.0))
            cv2.circle(layer, pt, r, c, -1, cv2.LINE_AA)
            cv2.circle(layer, pt, r + 2, (255, 255, 255) if tip else c, 1,
                       cv2.LINE_AA)

        glow = cv2.GaussianBlur(layer, (0, 0), 7)
        roi = out[y0:y1, x0:x1]
        res = cv2.add(roi, layer)
        res = cv2.add(res, glow)
        res = cv2.add(res, glow)       # stronger neon bloom
        out = out.copy() if out is not None else out
        out[y0:y1, x0:x1] = res
        return out

    def _silhouette_points(self, mask_s: np.ndarray, max_pts: int = 220):
        """Sample random points inside the person mask (full-res coords)."""
        ys, xs = np.nonzero(mask_s > 0.4)
        if len(xs) == 0:
            return None
        idx = self._rng.choice(len(xs), size=min(max_pts, len(xs)),
                               replace=False)
        pts = np.stack([(xs[idx] + 0.5) * self.sx,
                        (ys[idx] + 0.5) * self.sy], axis=1).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, self.w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, self.h - 1)
        return pts

    def _spawn(self, pts, vel, life_lo, life_hi, size_lo, size_hi):
        n = len(pts)
        life = self._rng.uniform(life_lo, life_hi, n).astype(np.float32)
        size = self._rng.uniform(size_lo, size_hi, n).astype(np.float32)
        color = _PALETTE[self._rng.integers(0, len(_PALETTE), n)]
        self.particles.spawn(pts.astype(np.float32),
                             vel.astype(np.float32), life, size, color)

    def _wave(self, wipe_y: float) -> np.ndarray:
        """1 below the wave front (rows with y > wipe_y), 0 above; ragged."""
        return np.clip(
            (self.yy_s - wipe_y + (self.noise_s - 0.5) * self.NOISE_AMP)
            / self.BAND, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # effect renderers
    # ------------------------------------------------------------------ #
    def _render_disappearing(self, frame, background, removal_s, solid_s):
        p = min(1.0, self.state_t / self.DISAPPEAR_DUR)

        # The wave front travels bottom -> top. It starts below the picture
        # and ends above it (margin = noise + band) so the sequence begins
        # with the person fully visible and ends with them fully gone.
        margin = self.NOISE_AMP / 2 + self.BAND
        wipe_y = (self.h + margin) + (-margin - (self.h + margin)) * p
        gone = self._wave(wipe_y)
        # Body turns slightly translucent before the wave reaches it.
        pre = 0.25 * float(_smoothstep(p * 4.0))
        amount = np.clip(gone + (1.0 - gone) * pre, 0.0, 1.0)
        out = self._composite(frame, background, removal_s * amount)

        # Phase 1: golden glow ignites around the silhouette, then fades.
        glow = max(0.0, 1.0 - p * 2.2)
        if glow > 0.0:
            edge = np.clip(cv2.dilate(solid_s, self._k3)
                           - cv2.erode(solid_s, self._k3), 0.0, 1.0)
            edge = cv2.GaussianBlur(edge, (5, 5), 0)
            edge = cv2.resize(edge, (self.w, self.h),
                              interpolation=cv2.INTER_LINEAR)
            g8 = cv2.convertScaleAbs(edge, alpha=255.0 * glow * 0.85)
            tint = cv2.merge([cv2.convertScaleAbs(g8, alpha=0.25),
                              cv2.convertScaleAbs(g8, alpha=0.72), g8])
            out = cv2.add(out, tint)

        # Phase 3: spawn embers along the wave front, drifting upward.
        rows = np.nonzero(np.abs(self.yy_s[:, 0] - wipe_y) < 45.0)[0]
        if len(rows):
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            ys, xs = np.nonzero(solid_s[r0:r1] > 0.35)
            if len(xs):
                take = min(36, len(xs))
                idx = self._rng.choice(len(xs), size=take, replace=False)
                pts = np.stack([(xs[idx] + 0.5) * self.sx,
                                (ys[idx] + r0 + 0.5) * self.sy],
                               axis=1).astype(np.float32)
                vel = np.stack([
                    self._rng.uniform(-70, 70, take),
                    self._rng.uniform(-190, -60, take),
                ], axis=1).astype(np.float32)
                self._spawn(pts, vel, 0.45, 0.95, 2.0, 5.0)
        return out

    def _render_reappearing(self, frame, background, removal_s, solid_s):
        q = min(1.0, self.state_t / self.REAPPEAR_DUR)
        # Reveal sweeps top -> bottom: rows above the front show the person.
        margin = self.NOISE_AMP / 2 + self.BAND
        wipe_y = -margin + (self.h + 2 * margin) * q
        shown = 1.0 - self._wave(wipe_y)       # 1 above the front
        shown = np.clip(shown, 0.0, 1.0)
        out = self._composite(frame, background, removal_s * (1.0 - shown))

        # Particles stream inward toward the silhouette.
        pts = self._silhouette_points(solid_s, max_pts=40)
        if pts is not None and q < 0.85:
            n = len(pts)
            ang = self._rng.uniform(0, 2 * math.pi, n)
            dist = self._rng.uniform(40, 110, n)
            start = pts + np.stack([np.cos(ang), np.sin(ang)],
                                   axis=1) * dist[:, None]
            start[:, 0] = np.clip(start[:, 0], 0, self.w - 1)
            start[:, 1] = np.clip(start[:, 1], 0, self.h - 1)
            direction = pts - start
            norms = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
            speed = self._rng.uniform(160, 300, n)[:, None]
            vel = direction / norms * speed
            self._spawn(start, vel.astype(np.float32), 0.35, 0.6, 2.0, 4.5)
        return out

    def _render_glitch(self, frame: np.ndarray, p: float) -> np.ndarray:
        out = frame.copy()
        # RGB channel split wobble.
        shift = int(16 * math.sin(p * math.pi))
        if shift:
            b, g, r = cv2.split(frame)
            r = np.roll(r, shift, axis=1)
            b = np.roll(b, -shift, axis=1)
            out = cv2.merge([b, g, r])
        # Horizontal slice displacement.
        for _ in range(7):
            y0 = int(self._rng.integers(0, max(1, self.h - 24)))
            hh = int(self._rng.integers(8, 28))
            dx = int(self._rng.integers(-48, 48))
            if dx:
                out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], dx, axis=1)
        # Scanlines + slight flicker.
        out[::4] = cv2.convertScaleAbs(out[::4], alpha=0.82)
        flicker = 0.92 + 0.08 * math.sin(p * 40.0)
        return cv2.convertScaleAbs(out, alpha=flicker)
