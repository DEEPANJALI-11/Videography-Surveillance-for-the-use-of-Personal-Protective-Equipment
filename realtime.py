
import cv2
from ultralytics import YOLO

# Load the trained YOLO model (update the path to your model)
model = "/content/drive/MyDrive/Dataset/YOLO_Training/yolov8_model/weights/best.pt"  

# Start video capture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame, conf=0.3)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show the frame with PPE detection
    cv2.imshow("Real-Time PPE Detection (Press Q to Quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


