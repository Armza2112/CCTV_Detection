import cv2
import json
import time 
import os 
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response, send_file, make_response
from dotenv import load_dotenv
from pathlib import Path
import onnxruntime as ort
import threading

# --- Load Environment ---
base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883)) 
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
CAMERA_PORT = os.getenv("CAMERA_PORT", 0)

if isinstance(CAMERA_PORT, str) and CAMERA_PORT.isdigit():
    CAMERA_PORT = int(CAMERA_PORT)

app = Flask(__name__)

# --- Load ONNX Model (แทนที่ YOLO .pt) ---
model_path = base_dir.parent / "train_model" / "best.onnx"
# providers=['CPUExecutionProvider'] สำหรับ Raspberry Pi
session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Global Variables
last_frame_path = os.path.join(base_dir, "latest_detect.jpg")
person_count = 0
target_status = "OFF"
CHECK_INTERVAL = 2  # วินาที (ปรับตามใจชอบ ยิ่งเลขมาก Pi ยิ่งเย็น)

# --- AI Snapshot Logic ---
def detection_job():
    global person_count, target_status
    last_seen_time = 0
    off_delay = 5

    while True:
        cap = cv2.VideoCapture(CAMERA_PORT)
        success, frame = cap.read()
        if success:
            h_orig, w_orig = frame.shape[:2]
            
            # 1. Pre-process (ทำรูปให้เป็น 640x640 ตามที่โมเดลต้องการ)
            input_size = 640
            img = cv2.resize(frame, (input_size, input_size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # 2. Inference
            outputs = session.run(None, {input_name: img})
            preds = np.squeeze(outputs).T 

            # 3. Post-process (คำนวณกรอบให้ "ไม่เพี้ยน")
            boxes, confs = [], []
            for i in range(len(preds)):
                scores = preds[i][4:]
                conf = scores.max()
                if conf > 0.4: # Confidence Threshold
                    xc, yc, w, h = preds[i][:4]
                    # แปลงพิกัด 640x640 กลับเป็นพิกัดจริงของภาพกล้อง
                    x1 = int((xc - w/2) * (w_orig / input_size))
                    y1 = int((yc - h/2) * (h_orig / input_size))
                    nw = int(w * (w_orig / input_size))
                    nh = int(h * (h_orig / input_size))
                    boxes.append([x1, y1, nw, nh])
                    confs.append(float(conf))

            indices = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.5)
            person_count = len(indices)

            # 4. Drawing & Status
            current_time = time.time()
            if person_count > 0:
                last_seen_time = current_time
                target_status = "ON"
                for i in indices.flatten() if hasattr(indices, 'flatten') else indices:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                if current_time - last_seen_time > off_delay:
                    target_status = "OFF"
                else:
                    target_status = "ON"

            cv2.putText(frame, f"Count: {person_count} | Status: {target_status}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 5. Save Image (Atomic Write เพื่อไม่ให้รูปพัง)
            success_enc, buffer = cv2.imencode('.jpg', frame)
            if success_enc:
                temp_file = last_frame_path + ".tmp"
                with open(temp_file, "wb") as f:
                    f.write(buffer.tobytes())
                os.replace(temp_file, last_frame_path)

            print(f"[{time.ctime()}] Check Done. Status: {target_status}")
            
        cap.release() # ปิดกล้องทุกครั้งเพื่อคืนทรัพยากร
        time.sleep(CHECK_INTERVAL)

# --- Flask Web ---
@app.route('/')
def index():
    return f"""
    <html>
        <head>
            <title>CCTV Snapshot</title>
            <meta http-equiv="refresh" content="{CHECK_INTERVAL}">
            <style>
                body {{ background: #1a1a1a; color: white; text-align: center; font-family: sans-serif; }}
                img {{ max-width: 90%; border: 4px solid #00ff00; border-radius: 10px; }}
            </style>
        </head>
        <body>
            <h1>Office AI (Snapshot Mode)</h1>
            <h2>Status: {target_status} | People: {person_count}</h2>
            <img src="/last_image?t={time.time()}">
            <p>Last update: {time.ctime()}</p>
        </body>
    </html>
    """

@app.route('/last_image')
def last_image():
    if not os.path.exists(last_frame_path):
        return Response("Loading...", status=404)
    response = make_response(send_file(last_frame_path, mimetype='image/jpeg'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

if __name__ == "__main__":
    threading.Thread(target=detection_job, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)