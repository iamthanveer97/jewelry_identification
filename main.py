from hand_tracking import HandTracker
import cv2

cap = cv2.VideoCapture(0)  # or your video file
tracker = HandTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame, hands = tracker.detect_hands(frame)

    for hand in hands:
        print(f"Detected {hand['type']} hand with {len(hand['landmarks'])} landmarks")

    cv2.imshow("Hand Tracking", output_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

from ring_detection import RingDetector

detector = RingDetector('path_to_your_trained_model.pt')
cap = cv2.VideoCapture("test_videos/sample.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_rings(frame)
    frame_with_boxes = detector.draw_detections(frame, results)

    cv2.imshow("Ring Detection", frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

