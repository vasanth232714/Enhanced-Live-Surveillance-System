import streamlit as st
import cv2
import torch
from torchvision import models, transforms
import numpy as np
import threading
from playsound import playsound
import csv
from datetime import datetime

# Load pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Transform for image preprocessing
transform = transforms.Compose([transforms.ToTensor()])

# Log file setup
log_file = "detection_log.csv"
with open(log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Time", "Message"])  # Write headers

# Function to log detection
def log_detection(message):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([date, time, message])

# Function to play a warning sound
def play_warning_sound():
    def sound_thread():
        playsound("D:\\inlustro_object_detection\\alarm\\burglar_alarm.mp3")  # Replace with your alarm sound file path
    threading.Thread(target=sound_thread, daemon=True).start()

# Function to detect objects
def detect_person(frame, confidence_threshold=0.5, roi=None):
    input_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)[0]

    person_detected = False
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score >= confidence_threshold and COCO_INSTANCE_CATEGORY_NAMES[label.item()] == "person":
            x1, y1, x2, y2 = map(int, box)

            # Check if the person is inside the ROI
            if roi is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi
                if (x1 < roi_x2 and x2 > roi_x1 and y1 < roi_y2 and y2 > roi_y1):  # Overlapping with ROI
                    person_detected = True
                    cv2.putText(frame, "WARNING: Person Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, person_detected

# Streamlit app
st.title("Enhanced Live Surveillance System")
st.write("Monitor a restricted area using live webcam feed with alerts and logging.")

start_surveillance = st.checkbox("Start Surveillance", key="start_surveillance")

if start_surveillance:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access the webcam. Please ensure it is connected.")
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = 200, 100, 400, 300  # Define ROI
        roi = (roi_x1, roi_y1, roi_x2, roi_y2)

        frame_placeholder = st.empty()
        stop_surveillance = st.checkbox("Stop Surveillance", key="stop_surveillance")

        while not stop_surveillance:
            ret, frame = cap.read()
            if not ret:
                st.warning("No frames received from the webcam.")
                break

            # Resize and process frame
            frame_resized = cv2.resize(frame, (640, 480))

            # Draw ROI on the frame
            cv2.rectangle(frame_resized, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
            cv2.putText(frame_resized, "Restricted Area", (roi_x1, roi_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Detect persons and trigger warning
            frame_with_boxes, person_detected = detect_person(frame_resized, roi=roi)
            if person_detected:
                play_warning_sound()
                log_detection("Person detected in the restricted area")

            # Display the frame
            frame_placeholder.image(frame_with_boxes, channels="BGR", use_column_width=True)

        cap.release()
        st.write("Surveillance stopped.")
        st.write(f"Detection logs saved in `{log_file}`.")
