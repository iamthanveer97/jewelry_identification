# main.py

import cv2
import sys
import os
from hand_tracking import HandTracker
from ring_detection import RingDetector

def is_ring_on_finger(ring_bbox, hand_landmarks, frame_width, frame_height):
    """
    Approximate logic: check if ring bbox is close to any finger joints
    """
    x1, y1, x2, y2 = ring_bbox
    ring_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    for hand in hand_landmarks:
        for x_norm, y_norm in hand:
            x = int(x_norm * frame_width)
            y = int(y_norm * frame_height)
            if abs(x - ring_center[0]) < 30 and abs(y - ring_center[1]) < 30:
                return True
    return False

def make_writer(video_path, frame_width, frame_height, fps=20.0,
                out_dir="output_test_video"):
    """
    Returns an opened cv2.VideoWriter that writes an MP4 file called
    <inputName>_output.mp4 inside *out_dir* (created if missing).
    """
    # 1. build output file name …/output_test_video/<input>_output.mp4
    base = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{base}_output.mp4")

    # 2. MPEG-4 codec that’s available on most default OpenCV builds
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")       # or *"avc1" / "H264" if you have them

    # 3. open writer and hand it back
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError("Couldn’t open VideoWriter - check codec/ffmpeg build")
    return writer, output_path

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read the first frame.")
        return

    frame_height, frame_width = frame.shape[:2]

    # ✅ Now it's safe to create VideoWriter
    # output_path = "output.avi"
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 20.0   # fallback if FPS not stored

    out, output_path = make_writer(video_path, frame_width, frame_height, fps)

    print(f"🔹 Writing processed frames to {output_path}")

    # Initialize detectors
    ring_detector = RingDetector(model_path="yolo_model/best.pt", conf_threshold=0.3)
    hand_tracker = HandTracker()

    while True:
        # You already read the first frame, so use it first
        results_frame = frame.copy()

        hand_landmarks = hand_tracker.detect_hands(results_frame)
        ring_detections = ring_detector.detect_rings(results_frame)

        for det in ring_detections:
            bbox = det["bbox"]
            if is_ring_on_finger(bbox, hand_landmarks, frame_width, frame_height):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(results_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(results_frame, "Ring", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for hand in hand_landmarks:
            for x_norm, y_norm in hand:
                x = int(x_norm * frame_width)
                y = int(y_norm * frame_height)
                cv2.circle(results_frame, (x, y), 3, (255, 0, 0), -1)

        # out.write(results_frame)
        # cv2.imshow("Ring + Hand Detection", results_frame)

        out.write(results_frame)
        cv2.imshow("Ring + Hand Detection", results_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_videos/sample.mp4"
    main(video_path)

