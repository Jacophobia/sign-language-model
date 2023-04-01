import cv2
import mediapipe as mp

import pandas as pd
import numpy as np

import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class HandTracker:
    def __init__(self, show_image=True):
        self.show_image = show_image

    def run(self, training=False, symbol=None, duration=5.0) -> pd.DataFrame:
        start_time = cv2.getTickCount()

        # create a dataframe to hold all captured data
        # labels: 'x1', 'y1', 'z1', ... 'x11', 'y11', 'z11', 'symbol'
        columns = []
        for i in range(1, 22):
            columns.append(f"x{i}")
            columns.append(f"y{i}")
            columns.append(f"z{i}")
        hand_data = pd.DataFrame(columns=columns + ["'symbol'"])

        self.cap = cv2.VideoCapture(0)
        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        if training:
                            # we want to store the x, y, z coordinates of each landmark as well as the symbol
                            row = np.array([symbol])
                            for landmark in hand_landmarks.landmark:
                                row = np.append(row, [landmark.x, landmark.y, landmark.z])
                            hand_data = pd.concat([hand_data, pd.DataFrame([row], columns=["'symbol'"] + columns)])

                # Flip the image horizontally for a selfie-view display.
                if self.show_image:
                    cv2.imshow("Hand Tracker", cv2.flip(image, 1))

                if (cv2.waitKey(5) & 0xFF == ord("q")) or (training and (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > duration):
                    break
        self.cap.release()

        return hand_data

if __name__ == '__main__':
    hand_tracker = HandTracker(show_image=True)
    hand_data = hand_tracker.run(training=True, symbol="?", duration=15.0)
    hand_data.to_csv(f"testing_data_abcs.csv", index=False)

    # letters_and_numbers = "ABCDEFGHIKLMNOPQRSTUVWXY"
    # duration = 10.0
    # for i in range(10):
    #     for letter in letters_and_numbers:
    #         print(f"Please show the letter/number {letter} for {duration} seconds")
    #         print("Please slowly move your hand around the screen to capture all possible positions")
    #         print("Also feel free to rotate your hand slightly to capture all possible angles")
    #         time.sleep(1)
    #         print("3")
    #         time.sleep(1)
    #         print("2")
    #         time.sleep(1)
    #         print("1")
    #         hand_data = hand_tracker.run(training=True, symbol=letter, duration=duration)
    #         hand_data.to_csv(f"training_data_{i}/{letter}.csv", index=False)
