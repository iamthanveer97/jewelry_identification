import math
import numpy as np

FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

_TOL_FRAC   = 0.40   # 40 % of larger bbox side
_MIN_TOL_PX = 8       # absolute floor (px)


def _lm_to_px(lms, w, h):
    return [(int(x * w), int(y * h)) for x, y in lms]

_AXIS_SEL = {
    "thumb": lambda p: (np.asarray(p[1], np.float32), np.asarray(p[2], np.float32)),
    "other": lambda p: (np.asarray(p[0], np.float32), np.asarray(p[1], np.float32)),
}

def _axis(name, pts):
    return _AXIS_SEL["thumb" if name == "thumb" else "other"](pts)


def _dist_pt_seg(p, a, b):
    ap, ab = p - a, b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        return float(np.linalg.norm(ap)), 0.0
    t = np.clip(np.dot(ap, ab) / ab2, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab))), float(t)


def find_ring_angle(bbox, hands, w, h):
    x1, y1, x2, y2 = bbox
    ring_h, ring_w = y2 - y1, x2 - x1
    tol_px = max(_TOL_FRAC * max(ring_h, ring_w), _MIN_TOL_PX)
    ring_c = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)

    best = None
    for name, idxs in FINGERS.items():
        for hand in hands:
            pts = _lm_to_px([hand[i] for i in idxs], w, h)
            a, b = _axis(name, pts)
            dist, t = _dist_pt_seg(ring_c, a, b)
            if dist < tol_px and 0.15 < t < 0.85:
                ang = round(-math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), 1)
                if best is None or dist < best[0]:
                    best = (dist, ang)
    return None if best is None else best[1]

# -----------------------------------------------------------------------------
#  Quick ring↔finger proximity (unchanged)  -----------------------------------
# -----------------------------------------------------------------------------

def is_ring_on_finger(bbox, hands, W, H):
    x1, y1, x2, y2 = bbox
    rcx, rcy = (x1 + x2) // 2, (y1 + y2) // 2
    for hand in hands:
        for xn, yn in hand:
            if abs(int(xn * W) - rcx) < 30 and abs(int(yn * H) - rcy) < 30:
                return True
    return False