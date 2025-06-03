# main.py
"""Ring-on-finger detector with inclination overlay

Adaptive tolerance variant
==========================
`tol_px = max( 0.40 × max(ring_h, ring_w), 8 px )`
so the radius grows with the larger side of the bbox but never shrinks below
8 pixels (enough to out-noise landmark jitter at 1080 p).
All other behaviour—blue dots, green "Ring" label, base-phalanx axis, ASCII
"deg"—is unchanged.
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

# -----------------------------------------------------------------------------
#  Video writer helper (unchanged)  -------------------------------------------
# -----------------------------------------------------------------------------

def make_writer(src, W, H, fps=20.0, out_dir="output_test_video"):
    base = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_output.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("Cannot open VideoWriter – check codec/ffmpeg build")
    return vw, out_path

# -----------------------------------------------------------------------------
#  Main loop  ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video", video_path)
        return

    ok, fr = cap.read()
    if not ok:
        print("❌ Empty video")
        return

    H, W = fr.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    out, dst = make_writer(video_path, W, H, fps)
    print("🔹 Writing to", dst)

    hand_tracker  = HandTracker()
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
