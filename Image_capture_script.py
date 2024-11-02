# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:04:36 2024

@author: Supriyo
"""

# import cv2
# import mediapipe as mp
# import os

# # Function to capture and save images for training
# import cv2
# import mediapipe as mp
# import os

# # Initialize Mediapipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# # Create a function to capture images
# def capture_images(label, num_images=100):
#     cap = cv2.VideoCapture(0)
#     img_count = 0
#     save_dir = os.path.join('dataset', label)

#     # Create directory if it does not exist
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     while img_count < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(frame_rgb)

#         if results.detections:
#             for detection in results.detections:
#                 # Draw the face detection annotations on the image.
#                 mp_drawing.draw_detection(frame, detection)
                
#                 # Save the image
#                 img_path = os.path.join(save_dir, f'{label}_{img_count}.jpg')
#                 cv2.imwrite(img_path, frame)
#                 img_count += 1
#                 print(f'Captured image {img_count} for {label}')
#                 break  # Capture one face at a time

#         # Display the frame
#         cv2.imshow('Capture Images', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Main script to capture images for different family members
# if __name__ == "__main__":
#     while True:
#         print("Enter the name of the person or 'q' to quit:")
#         label = input().strip()

#         if label.lower() == 'q':
#             break

#         print(f"Capturing images for {label}. Please look at the camera...")
#         capture_images(label)

#     print("Image capturing completed.")
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:04:36 2024

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
def capture_images(label, num_images=1000):
    cap = cv2.VideoCapture(0)
    img_count = 0
    save_dir = os.path.join('C:/Users/Supriyo/dataset', label)

    # Create directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Position yourself properly. Starting to capture images for {label} in 5 seconds...")
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
                print(f'Captured image {img_count} for {label}')
                break  # Capture one face at a time

        # Display the frame
        cv2.imshow('Capture Images', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script to capture images for different family members
if __name__ == "__main__":
    while True:
        print("Enter the name of the person or 'q' to quit:")
        label = input().strip()

        if label.lower() == 'q':
            break

        print(f"Capturing images for {label}. Please look at the camera...")
        capture_images(label)

    print("Image capturing completed.")


