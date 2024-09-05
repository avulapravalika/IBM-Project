from ultralytics import YOLO
import numpy as np
import cv2

# Load the model
model = YOLO("yolov8m.pt")

# Read video
cap = cv2.VideoCapture("road_trafifc.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)
    result = results[0]

    # Get bounding boxes and classes
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    # Draw bounding boxes and class labels
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    # Display the output
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)

    # Exit on pressing 'Esc'
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()