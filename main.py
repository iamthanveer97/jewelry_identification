# main.py

import cv2
import sys
import os

from hand_tracking import HandTracker
from ring_detection import RingDetector
from angle_utils import find_ring_angle, is_ring_on_finger


# -----------------------------------------------------------------------------
#  Video writer helper  -------------------------------------------
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
