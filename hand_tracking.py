# hand_tracking.py

import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_conf=0.5, tracking_conf=0.5): #These values can be played around for better hand prediction, and therefore ring prediction.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        hand_landmarks = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks: #Hand means different hand (and just hand as class)
                landmarks = [(lm.x, lm.y) for lm in hand.landmark] #Landmark contains exact ponts
                hand_landmarks.append(landmarks)
        return hand_landmarks  #Returns Multiple hand (Max 2) and 21 points per hand including joints, finger tip and wrist.

    def draw_hands(self, frame): #Mostly not used anywhere in the code
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
