import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained RNN model and label encoder
model = load_model("rnn_squat_posture_model.h5")

with open("rnn_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Mediapipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints using Mediapipe
def extract_keypoints_from_frame(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        keypoints = np.zeros((33, 4))  # Default if no landmarks are detected
    return np.array(keypoints).flatten()

# Open the webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution to 1600x900
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Extract keypoints from the frame
    keypoints = extract_keypoints_from_frame(frame)

    # Make predictions if keypoints are valid
    if np.any(keypoints):
        # Reshape keypoints for model input [samples, timesteps, features]
        keypoints_reshaped = keypoints.reshape(1, 1, -1)

        # Predict the posture
        prediction = model.predict(keypoints_reshaped)
        predicted_class = np.argmax(prediction, axis=1)
        posture_label = label_encoder.inverse_transform(predicted_class)[0]

        # Provide feedback
        if posture_label == "correct":
            feedback = "Good! Your squat form is correct. WWW"
        elif posture_label == "knees_in":
            feedback = "Keep your knees aligned with your toes!s"
        else:
            feedback = "Adjust your form!"

        # Display feedback on the frame
        cv2.putText(frame, f"Posture: {posture_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw pose landmarks
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow("Squat Posture Feedback", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


