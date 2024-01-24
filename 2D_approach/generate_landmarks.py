# generate_landmarks.py

import os
import cv2 as cv
import mediapipe as mp
from app import pre_process_landmark, logging_csv, KeyPointClassifier


def predict_landmarks(logkey, image_path, hands_model, keypoint_classifier_model):
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    image = cv.flip(image, 1)  # Mirror display

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands_model.process(image)
    image.flags.writeable = True

    # Process the results to get landmarks
    landmark_list = []
    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            # Landmark calculation
            landmark_list = calc_landmark_list(image, hand_landmarks)

    # Convert landmarks to the format expected by the keypoint_classifier
    pre_processed_landmark_list = pre_process_landmark(landmark_list)

    #Check if the landmark list is not empty before logging
    if pre_processed_landmark_list:
        # Write the predicted landmarks to the keypoint.csv file
        logging_csv(logkey, 1, pre_processed_landmark_list, [])

    return pre_processed_landmark_list



def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def main():
    # Load the trained models
    hands_model = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    keypoint_classifier_model = KeyPointClassifier()

    # Directory containing images for which you want to predict landmarks
    image_dir = "C:\\Git1\\hand-gesture-recognition-mediapipe\\DatasetForFingerCounter\\Validation"

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing image: {image_path}")
            logKey = 0
            if "zero" in filename:
                logKey = 0
            if "one" in filename:
                logKey = 1
            if "two" in filename:
                logKey = 2
            if "three" in filename:
                logKey = 3
            if "four" in filename:
                logKey = 4
            if "five" in filename:
                logKey = 5
            if "spock" in filename:
                logKey = 6
            if "other" in filename:
                logKey = 7
            # Predict landmarks for the current image
            predicted_landmarks = predict_landmarks(logKey,
                image_path, hands_model, keypoint_classifier_model
            )

            # Write the predicted landmarks to the keypoint.csv file
            # logging_csv(0, 1, predicted_landmarks, [])


if __name__ == '__main__':
    main()
