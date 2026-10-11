"""Decorative AI overlays drawn on top of the live (visible) person.

* Eye rays      - glowing irises that fire rotating, flickering light rays
* Heart reactor - an "arc reactor" on the chest that beats like a heart
* Boot sequence - when the system comes online: scan line, flash, shock-wave
                  and an "AI SYSTEM ONLINE" typewriter message

Everything is drawn additively into one layer (plus a cheap quarter-size blur
for the neon glow), so the cost is ~2-4 ms per frame. Overlays fade out
automatically whenever the person is not in the plain NORMAL state (vanishing,
hidden, glitch), so the invisible effect stays completely clean.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CYAN = (255, 225, 60)      # BGR light cyan
WHITE = (255, 255, 255)
BLUE = (255, 140, 30)


class AIOverlays:
    BOOT_DUR = 3.8
    # Where the "heart" sits relative to the chin (in face heights / widths).
    # Mirror view: your left side (heart) is on the left of the screen.
    HEART_DX = -0.38   # * face width  (negative = toward screen-left)
    HEART_DY = 1.05    # * face height below the chin

    def __init__(self, height: int, width: int):
        self.h, self.w = height, width
        self.eyes_on = True
        self.heart_on = True
        self.boot_t: float | None = None
        self._t = 0.0
        self._vis = 0.0
        self._pts: dict | None = None   # smoothed pixel geometry
        self._gw = max(8, width // 4)
        self._gh = max(8, height // 4)

    # ------------------------------------------------------------------ #
    def start_boot(self):
        self.boot_t = 0.0

    def reset(self):
        self.boot_t = None
        self._vis = 0.0
        self._pts = None

    # ------------------------------------------------------------------ #
    def _geometry(self, face) -> dict:
        w, h = self.w, self.h
        eyes = np.array(face["eyes"], dtype=np.float32) * [w, h]
        chin = np.array(face["chin"], dtype=np.float32) * [w, h]
        fore = np.array(face["forehead"], dtype=np.float32) * [w, h]
        fw = float(face["face_w"]) * w
        fh = float(np.linalg.norm(chin - fore))
        heart = np.array([chin[0] + self.HEART_DX * fw,
                          chin[1] + self.HEART_DY * fh], dtype=np.float32)
        return {"eyes": eyes, "heart": heart, "fw": np.float32(fw)}

    def draw(self, out: np.ndarray, face, dt: float, active: bool):
        """Return ``out`` with overlays added (copies before modifying)."""
        self._t += dt
        boot = None
        if self.boot_t is not None:
            if active:
                self.boot_t += dt
            if self.boot_t > self.BOOT_DUR:
                self.boot_t = None
            else:
                boot = self.boot_t

        # --- visibility / smoothing ------------------------------------ #
        if active and face is not None:
            g = self._geometry(face)
            if self._pts is None:
                self._pts = g
            else:
                for k in ("eyes", "heart", "fw"):
                    self._pts[k] = self._pts[k] + (g[k] - self._pts[k]) * 0.5
            self._vis = min(1.0, self._vis + dt * 5.0)
        else:
            self._vis = max(0.0, self._vis - dt * (6.0 if not active else 3.0))
            if self._vis <= 0.0:
                self._pts = None

        show_boot = boot is not None and active
        show_face = self._pts is not None and self._vis > 0.02
        if not show_face and not show_boot:
            return out

        layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        vis = self._vis
        burst = 0.0
        if boot is not None:
            # flash that peaks ~0.9 s into the boot sequence
            burst = math.exp(-((boot - 0.9) / 0.35) ** 2)

        if show_face:
            if self.eyes_on:
                for c in self._pts["eyes"]:
                    self._draw_eye(layer, c, float(self._pts["fw"]), vis, burst)
            if self.heart_on:
                self._draw_reactor(layer, self._pts["heart"],
                                   float(self._pts["fw"]), vis, boot)
        if show_boot:
            self._draw_boot(layer, boot)

        # neon glow: blur a quarter-size copy, add it back (cheap)
        small = cv2.resize(layer, (self._gw, self._gh),
                           interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(small, (0, 0), 2.2)
        glow = cv2.resize(glow, (self.w, self.h),
                          interpolation=cv2.INTER_LINEAR)
        res = cv2.add(out, layer)
        res = cv2.add(res, glow)
        return res

    # ------------------------------------------------------------------ #
    @staticmethod
    def _c(color, k):
        return tuple(int(max(0, min(255, v * k))) for v in color)

    def _draw_eye(self, layer, c, fw, vis, burst):
        cx, cy = int(c[0]), int(c[1])
        t = self._t
        k = vis * (1.0 + 0.8 * burst)
        r_core = max(2, int(0.028 * fw))
        cv2.circle(layer, (cx, cy), r_core, self._c(CYAN, k * 0.9), -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), max(1, r_core // 2), self._c(WHITE, k), -1,
                   cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), int(r_core * 2.2), self._c(CYAN, k * 0.7), 1,
                   cv2.LINE_AA)
        # rotating, flickering rays
        n = 14
        spin = t * 0.9
        for i in range(n):
            ang = spin + i * (2 * math.pi / n)
            flick = 0.55 + 0.45 * math.sin(t * 8.0 + i * 1.9)
            length = fw * (0.30 + 0.30 * flick) * (1.0 + 1.3 * burst)
            r0 = r_core * 2.2
            ca, sa = math.cos(ang), math.sin(ang)
            p0 = (int(cx + ca * r0), int(cy + sa * r0))
            pm = (int(cx + ca * (r0 + length * 0.45)),
                  int(cy + sa * (r0 + length * 0.45)))
            p1 = (int(cx + ca * (r0 + length)), int(cy + sa * (r0 + length)))
            cv2.line(layer, p0, pm, self._c(CYAN, k * 0.75), 1, cv2.LINE_AA)
            cv2.line(layer, pm, p1, self._c(BLUE, k * 0.5), 1, cv2.LINE_AA)
        # horizontal anamorphic lens streak
        half = int(fw * (0.55 + 0.9 * burst))
        cv2.line(layer, (cx - half, cy), (cx + half, cy),
                 self._c(CYAN, k * 0.7), 1, cv2.LINE_AA)

    def _draw_reactor(self, layer, c, fw, vis, boot):
        cx, cy = int(c[0]), int(c[1])
        if not (-200 < cx < self.w + 200 and -200 < cy < self.h + 200):
            return
        t = self._t
        # heartbeat: a sharp "lub-dub" about 72 bpm
        ph = (t * 1.2) % 1.0
        beat = (math.exp(-((ph - 0.05) / 0.05) ** 2)
                + 0.6 * math.exp(-((ph - 0.25) / 0.06) ** 2))
        R = fw * 0.34 * (1.0 + 0.16 * beat)
        k = vis * (0.9 + 0.5 * beat)
        cv2.circle(layer, (cx, cy), max(2, int(R * 0.40)), self._c(WHITE, k),
                   -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), int(R * 0.58), self._c(CYAN, k), 2,
                   cv2.LINE_AA)
        # rotating segmented ring
        base = (t * 80.0) % 360.0
        for i in range(8):
            a0 = base + i * 45.0
            cv2.ellipse(layer, (cx, cy), (int(R), int(R)), 0, a0, a0 + 30.0,
                        self._c(CYAN, k), 3, cv2.LINE_AA)
        # counter-rotating spokes
        for i in range(3):
            ang = math.radians(-t * 120.0 + i * 120.0)
            p0 = (int(cx + math.cos(ang) * R * 0.62),
                  int(cy + math.sin(ang) * R * 0.62))
            p1 = (int(cx + math.cos(ang) * R * 0.92),
                  int(cy + math.sin(ang) * R * 0.92))
            cv2.line(layer, p0, p1, self._c(BLUE, k), 2, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), int(R * 1.32 + beat * R * 0.15),
                   self._c(BLUE, k * 0.5), 1, cv2.LINE_AA)
        # boot shock-wave expanding from the chest
        if boot is not None and boot > 1.1:
            q = (boot - 1.1) / 1.2
            if q < 1.0:
                cv2.circle(layer, (cx, cy), int(R * (1.0 + 5.0 * q)),
                           self._c(CYAN, vis * (1.0 - q)), 2, cv2.LINE_AA)

    def _draw_boot(self, layer, boot):
        w, h = self.w, self.h
        # scan line sweeping top -> bottom during the first 1.6 s
        if boot < 1.6:
            y = int(h * boot / 1.6)
            cv2.line(layer, (0, y), (w, y), self._c(CYAN, 0.9), 2, cv2.LINE_AA)
            cv2.line(layer, (0, max(0, y - 6)), (w, max(0, y - 6)),
                     self._c(BLUE, 0.35), 1, cv2.LINE_AA)
        # typewriter text
        msg = "AI SYSTEM ONLINE"
        shown = msg[: int(min(1.0, max(0.0, (boot - 0.3) / 1.0)) * len(msg))]
        fade = 1.0 if boot < self.BOOT_DUR - 0.6 else \
            max(0.0, (self.BOOT_DUR - boot) / 0.6)
        if shown and fade > 0:
            size = cv2.getTextSize(shown, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
            x = (w - cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0][0]) // 2
            cv2.putText(layer, shown, (x, 52), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        self._c(CYAN, fade), 2, cv2.LINE_AA)
            if boot > 1.4:
                sub = "EYES SYNCED  |  HEART REACTOR ONLINE"
                s2 = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(layer, sub, ((w - s2[0]) // 2, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self._c(WHITE, fade * 0.85), 1, cv2.LINE_AA)
