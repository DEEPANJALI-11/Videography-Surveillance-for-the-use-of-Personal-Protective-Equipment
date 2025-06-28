from flask import Flask, render_template, Response, request, send_file
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Creating folders if not exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load YOLO model
model = YOLO("C:/Users/deepa/Desktop/Project/archive/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt")

# Webcam video generator function
def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3)
        annotated_frame = results[0].plot()

        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_ppe_detection(input_path):
    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Failed to read image file {input_path}")
    results = model(frame)
    annotated_frame = results[0].plot()
    
    output_path = os.path.join("outputs", os.path.basename(input_path))
    cv2.imwrite(output_path, annotated_frame)
    
    return output_path

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        input_path = os.path.join("uploads", file.filename)
        file.save(input_path)

        output_path = run_ppe_detection(input_path)

        return send_file(output_path, as_attachment=False)
    return "No file uploaded", 400

@app.route('/rules')
def rules():
    return render_template('rules.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
