# main.py
"""Ring-on-finger detector with inclination overlay (base-segment edition)

Change vs previous revision
===========================
* **_finger_axis** now uses the **proximal phalanx** (MCP → PIP) for
  index/middle/ring/pinky fingers so the ring is detected at the *base* of
  the finger where jewellery is normally worn.
* Thumb logic unchanged (still MCP → IP).
Everything else—tolerance, ASCII "deg", blue dots, green boxes—remains intact.
"""

import cv2
import sys
import os
import math
import numpy as np

from hand_tracking import HandTracker
from ring_detection import RingDetector

# -----------------------------------------------------------------------------
#  Geometry helpers  -----------------------------------------------------------
# -----------------------------------------------------------------------------

FINGERS = {
    "thumb":  [1, 2, 3, 4],      # landmarks inside thumb (keep MCP→IP)
    "index":  [5, 6, 7, 8],      # MCP–PIP–DIP–TIP
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

_MIN_TOL_PX = 12
_TOL_FRAC   = 0.40


def _lm_to_px(landmarks, w, h):
    return [(int(x * w), int(y * h)) for x, y in landmarks]


def _finger_axis(pts):
    """Return (A,B) describing the *base* segment.
    For thumb: MCP→IP (pts[1]→pts[2]) – unchanged.
    For other fingers: MCP→PIP (pts[0]→pts[1]).
    """
    if len(pts) != 4:
        return np.asarray(pts[0], np.float32), np.asarray(pts[1], np.float32)

    # Detect if this list is for thumb (landmark id 1 maps to pts[0])
    # Heuristic: thumb has x-coordinate much less than index when palm faces camera,
    # but simpler: pts belong to thumb if their original landmark ids start at 1.
    # We can pass thumb flag via call, but easier: thumb list comes from idxs of
    # FINGERS dict; we test by vertical order.
    # Simpler rule: if pts[0][0] < pts[1][0] and pts[0][1] > pts[1][1] maybe.
    # Rather than over-think, we rely on calling code: first entry in FINGERS is
    # thumb. Hence:
    #   – For thumb (passed via FINGERS["thumb"]) ⇒ use pts[1]→pts[2]
    #   – Else                                  ⇒ use pts[0]→pts[1]
    # The calling loop will pass thumb separately so we can know.
    return None  # placeholder to satisfy syntax; will be overwritten below

# Build a map of finger-name → axis-selector to avoid conditional per-point
_AXIS_CACHE = {
    "thumb": lambda p: (np.asarray(p[1], np.float32), np.asarray(p[2], np.float32)),
    "other": lambda p: (np.asarray(p[0], np.float32), np.asarray(p[1], np.float32)),
}

def _get_axis(name, pts):
    return _AXIS_CACHE["thumb" if name == "thumb" else "other"](pts)


def _dist_pt_seg(p, a, b):
    ap, ab = p - a, b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        return float(np.linalg.norm(ap)), 0.0
    t = np.clip(np.dot(ap, ab) / ab2, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest)), float(t)


def find_ring_angle(bbox, hands, w, h):
    x1, y1, x2, y2 = bbox
    ring_h = y2 - y1
    tol_px = max(ring_h * _TOL_FRAC, _MIN_TOL_PX)
    ring_c = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)

    best = None
    for name, idxs in FINGERS.items():
        for hand in hands:
            pts = _lm_to_px([hand[i] for i in idxs], w, h)
            a, b = _get_axis(name, pts)
            dist, t = _dist_pt_seg(ring_c, a, b)
            if dist < tol_px and 0.15 < t < 0.85:
                ang = round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), 1)
                if best is None or dist < best[0]:
                    best = (dist, ang)
    return None if best is None else best[1]

# -----------------------------------------------------------------------------
#  Quick proximity test (unchanged)  ------------------------------------------
# -----------------------------------------------------------------------------

def is_ring_on_finger(bbox, hands, W, H):
    x1, y1, x2, y2 = bbox
    rcx, rcy = (x1 + x2) // 2, (y1 + y2) // 2
    for hand in hands:
        for xn, yn in hand:
            if abs(int(xn * W) - rcx) < 30 and abs(int(yn * H) - rcy) < 30:
                return True
    return False

# -----------------------------------------------------------------------------
#  Writer helper (unchanged)  --------------------------------------------------
# -----------------------------------------------------------------------------

def make_writer(path, W, H, fps=20.0, out_dir="output_test_video"):
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_output.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("Cannot open VideoWriter – check codec/ffmpeg")
    return vw, out_path

# -----------------------------------------------------------------------------
#  Main loop  ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video", video_path)
        return

    ret, fr = cap.read()
    if not ret:
        print("❌ Empty video")
        return

    H, W = fr.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    out, dst = make_writer(video_path, W, H, fps)
    print("🔹 Writing to", dst)

    hand_tracker = HandTracker()
    ring_detector = RingDetector(model_path="yolo_model/best.pt")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hands = hand_tracker.detect_hands(frame)
        rings = ring_detector.detect_rings(frame)

        for det in rings:
            bb = det["bbox"]
            if not is_ring_on_finger(bb, hands, W, H):
                continue

            x1, y1, x2, y2 = bb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Ring", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            angle = find_ring_angle(bb, hands, W, H)
            if angle is not None:
                cv2.putText(frame, f"Angle: {angle} deg", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for hand in hands:
            for xn, yn in hand:
                cv2.circle(frame, (int(xn * W), int(yn * H)), 3, (255, 0, 0), -1)

        out.write(frame)
        cv2.imshow("Ring + Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "test_videos/sample.mp4"
    main(vid)
