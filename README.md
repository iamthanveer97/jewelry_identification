# jewelry_identification
Jewelry Ring Detection & Inclination Estimation

A lightweight computer-vision pipeline that detects rings on fingers in video and estimates the ring’s inclination angle using YOLOv8 and MediaPipe Hands.

⸻

Demo


⸻

How It Works
	1.	Ring detection – YOLOv8 (ring_detection.py) finds ring bounding-boxes in each frame.
	2.	Hand tracking – MediaPipe Hands (hand_tracking.py) returns 21 landmarks per detected hand.
	3.	Ring-to-Finger overlap – utils.is_ring_on_finger() checks whether any finger landmark lies inside a ring box.
	4.	Angle estimation – angle_utils.get_finger_angle() measures the 2-D angle between a finger’s base and tip; the angle is displayed near the ring.

⸻

1. Requirements

pip install -r requirements.txt

requirements.txt contains:
	•	ultralytics
	•	mediapipe
	•	opencv-python
	•	numpy (dependency of above)

⸻

2. Running the Project

# default test video
python main.py (searches for sample.mp4, user needs a sample video stored in /test_videos)

# your own video
python main.py test_videos/your_video.mp4

•	Press q to quit early.
•	Output is saved to test_videos/output_<video-name>.mp4.

⸻

3. Project Structure

jewelry_identification/
├── main.py                 # entry-point script
├── hand_tracking.py        # MediaPipe wrapper
├── ring_detection.py       # YOLOv8 wrapper
├── angle_utils.py          # finger-angle math
├── yolo_model/best.pt      # trained weights (tracked)
├── test_videos/            # input & output videos
│   └── .gitkeep
├── assets/                 # demo GIF / screenshots
├── requirements.txt
└── README.md


⸻

4. Output Example
	•	Green rectangle = detected ring
	•	Green text = finger name and inclination angle in degrees
	•	Blue skeleton = MediaPipe hand landmarks

⸻

5. Training Your Own Ring Detector
	1.	Collect & label ring images (Roboflow or CVAT).
	2.	Export in YOLOv8 format. (Google colab notebook to train the model is located in /yolo_model)
	3.	Train:

yolo task=detect mode=train model=yolov8n.pt data=your/data.yaml epochs=100 imgsz=640


	4.	Drop the resulting best.pt in yolo_model/.

⸻

6. Known Limitations / TODO
	•	Angle is 2-D; no depth estimation.
	•	Occasionally shows Angle: ? if no finger landmark overlaps the ring box (e.g. heavy occlusion).
	•	Currently supports rings; earrings/necklaces planned.

⸻

7. License

MIT License — see LICENSE for details.

⸻

8. Acknowledgements
	•	Ultralytics YOLOv8
	•	MediaPipe Hands
	•	Ring test videos courtesy of Pexels (free license)
    •	Roboflow, project id: jewelry-detection-esp.-ring
