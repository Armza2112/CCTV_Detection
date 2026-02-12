import cv2
import json
import time 
import os 
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response
from ultralytics import YOLO
from dotenv import load_dotenv
from pathlib import Path

# --- Load Environment ---
base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT")) 
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
CAMERA_PORT = os.getenv("CAMERA_PORT")

if CAMERA_PORT and CAMERA_PORT.isdigit():
    CAMERA_PORT = int(CAMERA_PORT)

# --- Initialize ---
app = Flask(__name__)
# model = YOLO("yolov8n.pt")
model = YOLO(r"C:\Users\cheew\Desktop\Swift\CCTV_Detected\train_model\best.pt")

# --- MQTT Setup ---
# client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# def mqtt_connect():
#     try:
#         if MQTT_USER and MQTT_PASS:
#             client.username_pw_set(MQTT_USER, MQTT_PASS)
#         client.connect(MQTT_BROKER, MQTT_PORT, 60)
#         client.loop_start()
#         print("Connected to MQTT Broker")
#     except Exception as e:
#         print(f"MQTT Connection Error: {e}")

# def mqtt_send(status):
#     payload = {"state": status}
#     client.publish(MQTT_TOPIC, json.dumps(payload))
#     print(f"Sent to MQTT: {status}")

# mqtt_connect()

# --- Logic สำหรับส่งภาพขึ้นเว็บ ---
def generate_frames():
    cap = cv2.VideoCapture(CAMERA_PORT)
    time.sleep(1)
    last_status = None
    last_seen_time = 0
    off_delay = 5

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_PORT)
            continue


        # 1. Detection
        results = model.predict(frame, classes=0, conf=0.4, verbose=False)
        person_count = len(results[0].boxes)
        
        # 2. ตีกรอบอัตโนมัติ (Bounding Boxes)
        annotated_frame = results[0].plot()

        # 3. Status & MQTT Logic
        current_time = time.time()
        if person_count > 0:
            last_seen_time = current_time
            target_status = "ON"
        else:
            if current_time - last_seen_time > off_delay:
                target_status = "OFF"
            else:
                target_status = "ON"

        # ส่ง MQTT เมื่อสถานะเปลี่ยนเท่านั้น
        # nonlocal last_status
        # if target_status != last_status:
        #     mqtt_send(target_status)
        #     last_status = target_status

        # 4. ตกแต่งภาพก่อนโชว์บนเว็บ
        cv2.putText(annotated_frame, f"People Count: {person_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Status: {target_status}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # 5. Encode เป็น JPEG สำหรับ Streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
