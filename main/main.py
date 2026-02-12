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

last_frame_path = os.path.join(base_dir, "latest_detect.jpg")
person_count = 0
target_status = "OFF"
CHECK_INTERVAL = 60 

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
    
    while True:
        cap = cv2.VideoCapture(CAMERA_PORT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        success, frame = cap.read()
        if success:
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
                # คำนวณ Scale ให้แม่นยำตามขนาดภาพจริง
                x1 = int((xc - w/2) * (w_orig / 640))
                y1 = int((yc - h/2) * (h_orig / 640))
                nw = int(w * (w_orig / 640))
                nh = int(h * (h_orig / 640))
                temp_boxes.append([x1, y1, nw, nh])
                confidences.append(float(max_scores[i]))

            final_indices = cv2.dnn.NMSBoxes(temp_boxes, confidences, 0.4, 0.5)
            person_count = len(final_indices)

            # วาดกรอบลงบนภาพจริง
            if person_count > 0:
                for i in (final_indices.flatten() if hasattr(final_indices, 'flatten') else final_indices):
                    x, y, w, h = temp_boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            curr_status = "ON" if person_count > 0 else "OFF"
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Detected: {person_count} | {timestamp_str}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # บันทึกไฟล์แบบไบต์ (แก้ปัญหา OpenCV imwrite)
            success_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success_enc:
                temp_file = last_frame_path + ".tmp"
                with open(temp_file, "wb") as f:
                    f.write(buffer.tobytes())
                os.replace(temp_file, last_frame_path)

            if curr_status != target_status:
                target_status = curr_status
                mqtt_send(target_status)

            print(f"[{timestamp_str}] AI Check: {person_count} found. Status: {target_status}")
        
        cap.release()
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def index():
    # ใช้ f-string เพื่อใส่ค่าปัจจุบันลงใน HTML โดยตรง
    return f"""
    <html>
        <head>
            <title>CCTV AI Snapshot</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ background: #1a1a1a; color: white; text-align: center; font-family: sans-serif; padding-top: 50px; }}
                img {{ max-width: 90%; border: 4px solid #00ff00; border-radius: 10px; box-shadow: 0 0 20px rgba(0,255,0,0.2); }}
                .info {{ font-size: 24px; color: #00ff00; margin-bottom: 20px; }}
                .time {{ color: #888; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Office Monitoring (Snapshot Mode)</h1>
            <div class="info">Current Status: {target_status} | People Count: {person_count}</div>
            <img src="/last_image?t={time.time()}">
            <div class="time">Last AI Scan: {time.ctime()}</div>
        </body>
    </html>
    """

@app.route('/last_image')
def last_image():
    if not os.path.exists(last_frame_path):
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "AI Initializing...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    # ส่งไฟล์และตั้งค่า Header ห้าม Cache
    response = make_response(send_file(last_frame_path, mimetype='image/jpeg'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

if __name__ == "__main__":
    threading.Thread(target=detection_job, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)