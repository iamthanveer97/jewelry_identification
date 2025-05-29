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
