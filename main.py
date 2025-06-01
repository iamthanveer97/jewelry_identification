# main.py

import cv2
import sys
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
    output_path = "output.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

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

