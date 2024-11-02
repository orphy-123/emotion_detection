# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:03:46 2024

@author: Supriyo
"""

import cv2
import mediapipe as mp
import os
import time

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Create a function to capture images
def capture_images(label, expression, num_images=1000):
    cap = cv2.VideoCapture(0)
    img_count = 0
    save_dir = os.path.join('C:/Users/Supriyo/dataset', expression, label)

    # Create directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Position yourself properly. Starting to capture images for {label} with expression {expression} in 5 seconds...")
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Starting in {i} seconds...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capture Images', frame)
        cv2.waitKey(1000)  # Wait for 1 second

    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                # Draw the face detection annotations on the image.
                mp_drawing.draw_detection(frame, detection)
                
                # Save the image
                img_path = os.path.join(save_dir, f'{label}_{img_count}.jpg')
                cv2.imwrite(img_path, frame)
                img_count += 1
                print(f'Captured image {img_count} for {label} with expression {expression}')
                break  # Capture one face at a time

        # Display the frame
        cv2.imshow('Capture Images', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script to capture images for different facial expressions
if __name__ == "__main__":
    expressions = ["Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust", "Neutral"]

    while True:
        print("Enter the name of the person or 'q' to quit:")
        label = input().strip()

        if label.lower() == 'q':
            break

        print("Select the expression to capture:")
        for i, expression in enumerate(expressions):
            print(f"{i + 1}. {expression}")

        expression_index = int(input("Enter the number corresponding to the expression: ")) - 1

        if 0 <= expression_index < len(expressions):
            selected_expression = expressions[expression_index]
            print(f"Capturing images for {label} with expression {selected_expression}. Please look at the camera...")
            capture_images(label, selected_expression)
        else:
            print("Invalid selection. Please try again.")

    print("Image capturing completed.")