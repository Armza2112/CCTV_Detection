import cv2
import json
import time 
import os 
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response
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

# Global variables สำหรับแชร์ข้อมูลระหว่าง Thread
current_frame = None
output_frame = None
detected_boxes = []
person_count = 0
target_status = "OFF"

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

def ai_thread():
    global detected_boxes, person_count, target_status, current_frame
    last_status = None
    last_seen_time = 0
    off_delay = 5

    while True:
        if current_frame is not None:
            img_frame = current_frame.copy()
            h_orig, w_orig = img_frame.shape[:2]
            
            # รัน AI ทุกๆ 0.1 - 0.2 วินาที (ไม่ให้ CPU ร้อนจัด)
            img = cv2.resize(img_frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            outputs = session.run(None, {input_name: img})
            preds = np.squeeze(outputs).T
            
            new_boxes = []
            scores = preds[:, 4:]
            max_scores = np.max(scores, axis=1)
            
            indices = np.where(max_scores > 0.4)[0]
            for i in indices:
                xc, yc, w, h = preds[i][:4]
                x1 = int((xc - w/2) * (w_orig / 640))
                y1 = int((yc - h/2) * (h_orig / 640))
                nw = int(w * (w_orig / 640))
                nh = int(h * (h_orig / 640))
                new_boxes.append([x1, y1, nw, nh])
            
            detected_boxes = new_boxes
            person_count = len(new_boxes)

            curr_time = time.time()
            if person_count > 0:
                last_seen_time = curr_time
                target_status = "ON"
            else:
                target_status = "OFF" if (curr_time - last_seen_time > off_delay) else "ON"

            if target_status != last_status:
                mqtt_send(target_status)
                last_status = target_status
            
        time.sleep(0.1) # ปรับค่านี้เพื่อคุมความเร็ว AI

def generate_frames():
    global current_frame, detected_boxes, person_count, target_status
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        current_frame = frame # ส่งเฟรมไปให้ AI Thread
        
        # วาดกรอบจากข้อมูลล่าสุดที่มี
        for box in detected_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"People: {person_count}  Status: {target_status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # เริ่มต้น AI Thread แยกต่างหาก
    t = threading.Thread(target=ai_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)