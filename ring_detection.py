# ring_detection.py

from ultralytics import YOLO
import cv2

class RingDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_rings(self, frame):
        """
        Detect rings in a given image frame using YOLOv8.
        
        Args:
            frame (numpy.ndarray): BGR image frame (from OpenCV)
        
        Returns:
            results: YOLO detection result (includes boxes, confidences, class ids)
        """
        results = self.model(frame)
        return results[0]  # return first result (usually only one frame is passed)

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes on the frame where rings are detected.
        
        Args:
            frame (numpy.ndarray): Original image
            results: Detection results from YOLO
        
        Returns:
            frame_with_boxes: Image with boxes drawn
        """
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            label = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Ring {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
