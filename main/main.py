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

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def mqtt_connect():
    try:
        if MQTT_USER and MQTT_PASS:
            client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"MQTT Error: {e}")

def mqtt_send(status):
    payload = {"state": status}
    client.publish(MQTT_TOPIC, json.dumps(payload))

mqtt_connect()

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_PORT)
    # --- จุดที่ 1: ลดความละเอียดกล้องตั้งแต่ต้นทาง ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    time.sleep(1)
    last_status = None
    last_seen_time = 0
    off_delay = 5
    
    # ตัวแปรสำหรับคุมความลื่น
    frame_count = 0
    person_count = 0
    target_status = "OFF"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_PORT)
            continue

        frame_count += 1
        # --- จุดที่ 2: ข้ามเฟรม รัน AI ทุกๆ 6 เฟรมพอ (ประมาณ 0.2 วินาทีครั้ง) ---
        if frame_count % 6 == 0:
            img = cv2.resize(frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            outputs = session.run(None, {input_name: img})
            preds = np.squeeze(outputs).T
            scores = preds[:, 4:]
            max_scores = np.max(scores, axis=1)
            person_count = np.sum(max_scores > 0.4) 

            current_time = time.time()
            if person_count > 0:
                last_seen_time = current_time
                target_status = "ON"
            else:
                target_status = "OFF" if (current_time - last_seen_time > off_delay) else "ON"

            if target_status != last_status:
                mqtt_send(target_status)
                last_status = target_status
            
            frame_count = 0 # reset ตัวนับ

        # วาดข้อความ (วาดทุกเฟรมเพื่อให้ตัวเลขไม่กระพริบ)
        cv2.putText(frame, f"People: {person_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {target_status}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)