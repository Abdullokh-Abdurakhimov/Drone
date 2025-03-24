import cv2
import torch
import numpy as np
import threading
from gtts import gTTS  # Google Text-to-Speech
import pygame  # For non-blocking audio playback
import os

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Initialize the tracker and Kalman filter
tracker = cv2.TrackerCSRT_create()

# Kalman Filter setup for tracking stability
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Set a flag to know if the tracker is initialized
tracker_initialized = False

# Get the camera frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define thresholds for movement and Kalman prediction
center_threshold = frame_width // 3  # For left/right movement
distance_threshold_small = frame_height // 4  # For forward movement
distance_threshold_large = frame_height // 2  # For backward movement

# Define a baseline height (the size of the object when it's at the ideal distance)
baseline_height = 200  # This can be adjusted based on the object and distance

# Variable to store frame
frame = None
target_name = None  # To hold the detected target name
previous_target = None  # To prevent repeated announcements

# Function to announce the target name using TTS in a separate thread
def announce_target(name):
    global previous_target
    if name != previous_target:
        # Run TTS in a separate thread to avoid blocking
        tts_thread = threading.Thread(target=speak, args=(name,))
        tts_thread.start()
        previous_target = name  # Update previous target

# TTS function to speak the target name using gTTS and pygame for non-blocking audio playback
def speak(name):
    tts = gTTS(text=f"Target Identified: {name}", lang='en')
    tts.save("target_announcement.mp3")
    
    # Play the audio file asynchronously using pygame
    pygame.mixer.music.load("target_announcement.mp3")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():  # Wait for the audio to finish playing
        continue

    os.remove("target_announcement.mp3")  # Remove the file after playback

# Function to handle object detection in a separate thread
def detect_objects():
    global frame, tracker_initialized, target_name

    while True:
        if not tracker_initialized and frame is not None:
            results = model(frame)
            if results.xyxy[0].shape[0] > 0:
                x1, y1, x2, y2, conf, cls = results.xyxy[0][0]
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                # Initialize tracker with bounding box
                tracker.init(frame, bbox)
                tracker_initialized = True
                
                # Get the object class label (name)
                target_name = results.names[int(cls)]
                print(f"Target Identified: {target_name}")  # Print target name in the console
                
                # Announce the target name using TTS
                announce_target(target_name)

# Start object detection in a separate thread
threading.Thread(target=detect_objects, daemon=True).start()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    if tracker_initialized:
        # Track the object using the CSRT tracker
        success, bbox = tracker.update(frame)
        if success:
            # Get the bounding box coordinates
            x, y, w, h = [int(v) for v in bbox]
            object_center_x = x + w // 2
            object_center_y = y + h // 2
            
            # Update Kalman filter with measured position
            measured = np.array([[np.float32(object_center_x)], [np.float32(object_center_y)]])
            kalman.correct(measured)
            
            # Predict the next position
            predicted = kalman.predict()
            predicted_center_x, predicted_center_y = int(predicted[0]), int(predicted[1])
            
            # Draw a rectangle around the tracked object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw the predicted position (as a small circle)
            cv2.circle(frame, (predicted_center_x, predicted_center_y), 5, (0, 0, 255), -1)
            
            # Display the target name above the rectangle
            if target_name is not None:
                cv2.putText(frame, f'Target: {target_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Determine movement direction based on predicted position
            if predicted_center_x < center_threshold:
                movement_direction = "Move Left"
            elif predicted_center_x > (frame_width - center_threshold):
                movement_direction = "Move Right"
            else:
                movement_direction = "Stay Center"
            
            # Estimate depth based on the object's height (larger height means closer)
            if h < distance_threshold_small:
                movement_direction += " | Move Forward"
            elif h > distance_threshold_large:
                movement_direction += " | Move Backward"
            else:
                movement_direction += " | Stay"
            
            # Adjust drone height based on object height
            if h < baseline_height:
                movement_direction += " | Move Up"
            elif h > baseline_height:
                movement_direction += " | Move Down"
            
            # Print drone movement commands
            print(f"Object Height: {h}, Drone Command: {movement_direction}")
    
    # Display the result (drone camera feed with detection, tracking, and target name)
    cv2.imshow('Autonomous Target Following with Target Name', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
