Design Document – Jewelry Ring Detection & Inclination Estimation

Author: Thanveer Ahamed
Repository: jewelry_identification/
Date: 03 June 2025

⸻

1  Objective

Detect rings on fingers in recorded video and annotate each with a 2‑D inclination angle, enabling fashion creators to showcase jewelry dynamically.

⸻

2  Scope & Assumptions

In‑Scope

    Rings on any finger (both hands)

    Offline MP4 processing

    2‑D in‑plane angle (no depth)

    Up to two hands visible (MediaPipe limit)

Out‑of‑Scope

    Earrings, necklaces, dress segmentation

    Real‑time mobile inference

    Full 3‑D pose or depth angle

    More than two hands

Assumptions: rings appear at reasonable resolution and are not heavily occluded.

⸻

3  Key Tasks & Library Choices 

Task (Library / Model)
-> Why this choice

a. Finger‑landmark tracking	( MediaPipe Hands )
    ->Suggested by ChatGPT; 21 landmarks, CPU‑friendly.

b. Ring detection	( YOLOv8 Ultralytics )
    ->Suggested by ChatGPT; fast to fine‑tune for small custom objects.

c. Ring–finger ( association Bounding‑box ∩ landmark test (angle_utils) )
    ->Arrived via intuition; minimal compute and generally reliable.

d. Angle estimation ( Vector from finger base (MCP) → proximal (PIP) )
    ->Arrived via intuition; more stable than using fingertip direction.


⸻

4  Pipeline Overview

graph TD
    A[Input .mp4] --> B[YOLOv8 Ring Detection]
    A --> C[MediaPipe Hand Landmarks (≤2 hands)]
    B --> D[Ring Tracking (per‑frame list)]
    C --> E[Ring ↔ Finger Association]
    D --> E
    E --> F[Angle Calculation (base→proximal)]
    F --> G[Overlay & VideoWriter]
    G --> H[output_<video>.mp4]


⸻

### 4.1  Dataset & YOLO Training
	•	76 base images → Roboflow.
	•	Augmentations (flip, resize) → ≈150 images.
	•	Train/val/test = 80 / 10 / 10.
	•	YOLOv8‑nano, 50 epochs @ 360×360 in Google Colab.
	•	best.pt stored in yolo_model/.

Metric	   (Value)
Precision	0.874
Recall	    0.975

### 4.2  Ring Tracking (Frame‑to‑Frame)
For v1 we reuse YOLO detections each frame; ID persistence is not critical.  Future versions may integrate Deep SORT for smoother temporal IDs.

### 4.3  Hand Tracking
	•	MediaPipe returns 21 normalized landmarks per detected hand.
	•	Library limit: maximum two hands per frame.
	•	Converted to pixel coords inside hand_tracking.py.

### 4.4  Ring–Finger Association
angle_utils.is_ring_on_finger() tests if any landmark of a finger falls inside the ring box.  If multiple fingers overlap, the nearest base joint wins.

### 4.5  Angle Estimation (base → proximal vector)
For the matched finger we look at:

Finger	base_idx (MCP)	prox_idx (PIP)
Thumb	 2 	                3 
Index	 5 	                6 
Middle	 9 	                10
Ring	 13	                14
Pinky	 17	                18

When a ring’s bounding-box overlaps a finger, the system needs a single, intuitive orientation. Because a ring sits at the base of a finger, the vector from the metacarpophalangeal (MCP) joint to the first proximal (PIP) joint is an excellent proxy for the ring’s tilt: it rotates only when the whole finger rotates and is unaffected by fingertip jitter.

Angle formula (pixel space):
theta = atan2(y_PIP - y_MCP, x_PIP - x_MCP)   # angle in radians
Value is rounded to one decimal and displayed as “Index Angle: 42.6 deg”.

### 4.6  Overlay & Output
main.py draws:
	•	Blue skeleton (MediaPipe).
	•	Green ring box.
	•	Green text angle.
	•	Saves MP4 (mp4v codec) to test_videos/output_<name>.mp4.

⸻

5  Known Limitations & Future Work

      Limitation	             Planned Improvement

a. Dataset still modest	   Collect more diverse ring images (lighting, skin tones, poses).

b. False positive when     Add depth cue / segmentation or stricter landmark overlap ratio.
   ring rests on finger 
   but is not worn	

c. Angle shows “?” when    Fallback: pick finger with smallest centroid distance.
   no overlap	

d. Only 2‑D orientation	   Investigate 3‑D hand mesh (MediaPipe Holistic) or depth sensor input.


⸻

6  Repro Steps

pip install -r requirements.txt          # deps
python main.py test_videos/sample.mp4    # run

Outputs annotated video to test_videos/output_sample.mp4.

⸻

7  Repository Tree

jewelry_identification/
├── main.py
├── hand_tracking.py
├── ring_detection.py
├── utils.py
├── angle_utils.py
├── yolo_model/best.pt
├── test_videos/
└── README.md


⸻

8  Acknowledgements
	•	Ultralytics YOLOv8
	•	MediaPipe Hands (Google)
	•	Roboflow
	•	ChatGPT – for surfacing these libraries and bootstrapping code snippets

⸻

Status: Core pipeline functional; future work will focus on larger datasets, 3‑D angles, and ring‑ID tracking across frames.