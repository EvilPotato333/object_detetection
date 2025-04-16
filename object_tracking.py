import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

# Load video file
cap = cv2.VideoCapture("just_a_video.mp4")

# Set frame width and height to a smaller size for faster processing
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)  # Reduce to 50% width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)  # Reduce to 50% height

# Resize video frames to smaller size for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

# Process every nth frame to speed up processing (e.g., every 2nd frame)
frame_skip = 2

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Skip frames to reduce processing load
    if count % frame_skip != 0:
        continue

    # Detect objects on frame
    detections = od.detect(frame)

    for (class_name, confidence, box) in detections:
        (x, y, w, h) = box

        # Draw bounding box, class name, and confidence score
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame with adjusted waitKey for real-time display
    cv2.imshow("Frame", frame)

    # Control the playback speed
    key = cv2.waitKey(1)  # Small delay for real-time playback
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
