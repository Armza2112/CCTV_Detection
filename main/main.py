import cv2
import json
import time 
import os 
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response, send_file
from dotenv import load_dotenv
from pathlib import Path
import onnxruntime as ort
import threading

base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT")) 
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
CAMERA_PORT = os.getenv("CAMERA_PORT", 0)

if isinstance(CAMERA_PORT, str) and CAMERA_PORT.isdigit():
    CAMERA_PORT = int(CAMERA_PORT)

app = Flask(__name__)
model_path = base_dir.parent / "train_model" / "best.onnx"

session = ort.InferenceSession(str(model_path))
input_name = session.get_inputs()[0].name

# Global Variables
last_frame_path = "latest_detect.jpg"
person_count = 0
target_status = "OFF"
CHECK_INTERVAL = 60 # วินาที (เปลี่ยนเป็น 60 คือ 1 นาที)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
def mqtt_connect():
    try:
        if MQTT_USER and MQTT_PASS:
            client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except: pass

def mqtt_send(status):
    client.publish(MQTT_TOPIC, json.dumps({"state": status}))

mqtt_connect()

def detection_job():
    global person_count, target_status
    cap = cv2.VideoCapture(CAMERA_PORT)
    
    while True:
        success, frame = cap.read()
        if success:
            # ทำ AI Detection
            h_orig, w_orig = frame.shape[:2]
            blob = cv2.resize(frame, (640, 640))
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            outputs = session.run(None, {input_name: blob})
            preds = np.squeeze(outputs).T
            
            temp_boxes = []
            confidences = []
            max_scores = np.max(preds[:, 4:], axis=1)
            indices = np.where(max_scores > 0.4)[0]

            for i in indices:
                xc, yc, w, h = preds[i][:4]
                x1 = int((xc - w/2) * (w_orig / 640))
                y1 = int((yc - h/2) * (h_orig / 640))
                nw = int(w * (w_orig / 640))
                nh = int(h * (h_orig / 640))
                temp_boxes.append([x1, y1, nw, nh])
                confidences.append(float(max_scores[i]))

            final_indices = cv2.dnn.NMSBoxes(temp_boxes, confidences, 0.4, 0.5)
            person_count = len(final_indices)

            # วาดกรอบและบันทึกภาพลงไฟล์
            for i in (final_indices.flatten() if len(final_indices) > 0 else []):
                x, y, w, h = temp_boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            curr_status = "ON" if person_count > 0 else "OFF"
            cv2.putText(frame, f"Detected: {person_count} | {time.ctime()}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imwrite(last_frame_path, frame)

            # ส่ง MQTT ถ้าสถานะเปลี่ยน
            if curr_status != target_status:
                target_status = curr_status
                mqtt_send(target_status)

            print(f"[{time.ctime()}] Check Done: {person_count} found.")

        # พักเครื่องตามเวลาที่กำหนด
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def index():
    # ส่งหน้าเว็บที่สั่งให้ Refresh รูปเองทุก 30 วินาที
    return """
    <html>
        <head><title>CCTV Snapshot</title><meta http-equiv="refresh" content="30"></head>
        <body style="background:#222; color:white; text-align:center;">
            <h1>CCTV Latest Detection</h1>
            <img src="/last_image" style="max-width:80%; border:5px solid #00ff00;">
            <p>Last Update: <span id="time"></span></p>
            <script>document.getElementById('time').innerHTML = new Date().toLocaleString();</script>
        </body>
    </html>
    """

@app.route('/last_image')
def last_image():
    return send_file(last_frame_path, mimetype='image/jpeg')

if __name__ == "__main__":
    # รัน AI ใน Background
    threading.Thread(target=detection_job, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)