import cv2
import mediapipe as mp

import numpy as np

import pickle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class HandTracker:
    def __init__(self, show_image=True):
        self.show_image = show_image
        self.model = pickle.load(open("sign_language_model.sav", "rb"))

    def get_letter(self, num) -> str:
        num_to_letter = {
            0: 'A', 
            1: 'B', 
            2: 'C', 
            3: 'D', 
            4: 'E', 
            5: 'F', 
            6: 'G', 
            7: 'H', 
            8: 'I', 
            9: 'K', 
            10: 'L', 
            11: 'M', 
            12: 'N', 
            13: 'O', 
            14: 'P', 
            15: 'Q', 
            16: 'R', 
            17: 'S', 
            18: 'T', 
            19: 'U', 
            20: 'V', 
            21: 'W', 
            22: 'X', 
            23: 'Y'
        }
        return num_to_letter[round(num)]


    def get_prediction(self, row) -> str:
        if len(row) != 63:
            return "?"
        # get the xyz coordinates of the points
        points = row.reshape(-1, 3)
        # get the center of the hand
        center = np.mean(points, axis=0)
        # get the distances between the points and the center
        distances = np.linalg.norm(points - center, axis=1)
        # add the distances to the normalized dataframe
        predictions = self.model.predict(distances.reshape(1, -1))
        if len(predictions) > 0:
            return self.get_letter(predictions[0])
        else:
            return "?"

    def run(self) -> None:
        self.cap = cv2.VideoCapture(0)
        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # To improve performance, mark the image as not writeable to
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
                        # we want to store the x, y, z coordinates of each landmark as well as the symbol
                        row = np.array([])
                        for landmark in hand_landmarks.landmark:
                            row = np.append(row, [landmark.x, landmark.y, landmark.z])
                        print(self.get_prediction(row))

                # Flip the image horizontally for a selfie-view display.
                if self.show_image:
                    cv2.imshow("Hand Tracker", cv2.flip(image, 1))

                if (cv2.waitKey(5) & 0xFF == ord("q")):
                    break
        self.cap.release()


if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()