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

# โหลด Model ครั้งเดียวเก็บไว้ใน RAM
session = ort.InferenceSession(str(model_path))
input_name = session.get_inputs()[0].name

# ระบุ Path ไฟล์รูปให้ชัดเจน ป้องกันหาไฟล์ไม่เจอ
last_frame_path = os.path.join(base_dir, "latest_detect.jpg")
person_count = 0
target_status = "OFF"
CHECK_INTERVAL = 60 # ตรวจสอบทุกๆ 60 วินาที

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def mqtt_connect():
    try:
        if MQTT_USER and MQTT_PASS:
            client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except:
        pass

def mqtt_send(status):
    client.publish(MQTT_TOPIC, json.dumps({"state": status}))

mqtt_connect()

def detection_job():
    global person_count, target_status
    
    while True:
        cap = cv2.VideoCapture(CAMERA_PORT)
        # ตั้งค่ากล้องให้ถอดรหัลง่ายๆ
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        success, frame = cap.read()
        if success:
            h_orig, w_orig = frame.shape[:2]
            
            # ประมวลผล AI
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

            # วาดกรอบลงในรูป
            if person_count > 0:
                for i in (final_indices.flatten() if hasattr(final_indices, 'flatten') else final_indices):
                    x, y, w, h = temp_boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # บันทึกสถานะและส่ง MQTT
            curr_status = "ON" if person_count > 0 else "OFF"
            cv2.putText(frame, f"Detected: {person_count} | {time.ctime()}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # เขียนไฟล์แบบปลอดภัย (Atomic Write)
            temp_file = last_frame_path + ".tmp"
            cv2.imwrite(temp_file, frame)
            os.replace(temp_file, last_frame_path)

            if curr_status != target_status:
                target_status = curr_status
                mqtt_send(target_status)

            print(f"[{time.ctime()}] AI Check: {person_count} person(s) found. Status: {target_status}")
        
        cap.release() # ปิดกล้องเพื่อประหยัดพลังงานและคืนทรัพยากร
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def index():
    # หน้าเว็บแบบเรียบง่าย สั่ง Refresh รูปทุก 30 วินาที
    return """
    <html>
        <head>
            <title>CCTV AI Snapshot</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { background: #1a1a1a; color: white; text-align: center; font-family: sans-serif; }
                img { max-width: 90%; border: 4px solid #444; border-radius: 8px; margin-top: 20px; }
                .status { font-size: 24px; color: #00ff00; }
            </style>
        </head>
        <body>
            <h1>CCTV AI Snapshot (Every 1 Min)</h1>
            <div class="status">Current Status: {{ status }}</div>
            <img src="/last_image?t={{ time }}">
            <p>Last Update: {{ last_time }}</p>
        </body>
    </html>
    """.replace('{{ status }}', target_status).replace('{{ last_time }}', time.ctime()).replace('{{ time }}', str(time.time()))

@app.route('/last_image')
def last_image():
    # ถ้ายังไม่มีไฟล์รูป ให้ส่งรูปว่างๆ ไปก่อน ป้องกัน Error 500
    if not os.path.exists(last_frame_path):
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "AI Initializing...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    return send_file(last_frame_path, mimetype='image/jpeg')

if __name__ == "__main__":
    # เริ่มระบบตรวจจับใน Thread แยก
    threading.Thread(target=detection_job, daemon=True).start()
    # รัน Flask Web Server
    app.run(host='0.0.0.0', port=5000, debug=False)