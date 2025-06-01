# ring_detection.py

from ultralytics import YOLO
import cv2

class RingDetector:
    def __init__(self, model_path="models/best.pt", conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_rings(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0])
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "class_id": cls_id
                })
        return detections
