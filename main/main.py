import cv2
import json
import time 
import os 
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response, send_file, make_response
from ultralytics import YOLO  # ใช้ตัวเดิมที่คุ้นเคย
from dotenv import load_dotenv
from pathlib import Path
import threading

# --- Setup ---
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")

# ระบุ Path ไปที่ .pt โดยตรง
model_path = base_dir.parent / "train_model" / "best.pt"
model = YOLO(str(model_path))

app = Flask(__name__)
last_frame_path = os.path.join(base_dir, "latest_detect.jpg")
person_count = 0
target_status = "OFF"
CHECK_INTERVAL = 2 # วินาที (ถ้า Pi ร้อนไปให้เพิ่มเป็น 3-5)

def detection_job():
    global person_count, target_status
    last_seen_time = 0
    off_delay = 5
    CAMERA_PORT = os.getenv("CAMERA_PORT", 0)
    if str(CAMERA_PORT).isdigit(): CAMERA_PORT = int(CAMERA_PORT)

    while True:
        cap = cv2.VideoCapture(CAMERA_PORT)
        # ตั้งค่าความละเอียดให้ไม่สูงเกินไป เพื่อให้ Pi ประมวลผลเร็วขึ้น
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        success, frame = cap.read()
        if success:
            # 1. Inference ด้วย .pt (ใช้สูตรเดียวกับใน Windows)
            results = model.predict(frame, conf=0.4, verbose=False)
            person_count = len(results[0].boxes)
            
            # 2. ตีกรอบ (วิธีนี้กรอบจะตรงเป๊ะแน่นอน ไม่มั่วสเกล)
            annotated_frame = results[0].plot()

            # 3. Status Logic
            current_time = time.time()
            if person_count > 0:
                last_seen_time = current_time
                target_status = "ON"
            else:
                target_status = "OFF" if (current_time - last_seen_time > off_delay) else "ON"

            # 4. ใส่ข้อความเสริม
            cv2.putText(annotated_frame, f"AI Snapshot | Count: {person_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 5. บันทึกรูป (Atomic Write)
            success_enc, buffer = cv2.imencode('.jpg', annotated_frame)
            if success_enc:
                temp_file = last_frame_path + ".tmp"
                with open(temp_file, "wb") as f:
                    f.write(buffer.tobytes())
                os.replace(temp_file, last_frame_path)

            print(f"[{time.strftime('%H:%M:%S')}] AI Processed. Found: {person_count}")

        cap.release() # คืนค่ากล้องทุกรอบ ป้องกัน device busy
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def index():
    return f"""
    <html>
        <head>
            <title>AI Snapshot Pi</title>
            <meta http-equiv="refresh" content="{CHECK_INTERVAL}">
            <style>
                body {{ background: #111; color: #0f0; text-align: center; font-family: monospace; }}
                img {{ max-width: 85%; border: 2px solid #333; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h2>AI STATUS: {target_status} | COUNT: {person_count}</h2>
            <img src="/last_image?t={time.time()}">
            <p>Next update in {CHECK_INTERVAL}s...</p>
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
    app.run(host='0.0.0.0', port=5000, debug=False)