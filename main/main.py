import cv2
import time
import os
from flask import Flask, Response
from ultralytics import YOLO
from dotenv import load_dotenv
from pathlib import Path

# --- Setup & Config ---
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")

# ดึงค่ากล้องจาก .env (ถ้าไม่มีจะใช้เลข 0)
CAMERA_PORT = os.getenv("CAMERA_PORT", 0)
if str(CAMERA_PORT).isdigit(): 
    CAMERA_PORT = int(CAMERA_PORT)

model_path = base_dir.parent / "train_model" / "best_ncnn_model"
model = YOLO(str(model_path))

app = Flask(__name__)

# ตัวแปร Global สำหรับสถานะ
person_count = 0
target_status = "OFF"
last_seen_time = 0
off_delay = 5 

def generate_frames():
    global person_count, target_status, last_seen_time
    
    # เปิดกล้องค้างไว้ (Real-time ต้องไม่ release ใน loop)
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Inference: ปรับ imgsz=320 เพื่อความลื่นไหลบน Pi
        results = model.predict(frame, conf=0.4, imgsz=320, verbose=False, stream=True)
        
        current_count = 0
        for r in results:
            current_count = len(r.boxes)
            frame = r.plot() # วาดกรอบอัตโนมัติ

        # Logic: จัดการสถานะ ON/OFF
        person_count = current_count
        current_time = time.time()
        if person_count > 0:
            last_seen_time = current_time
            target_status = "ON"
        else:
            target_status = "OFF" if (current_time - last_seen_time > off_delay) else "ON"

        # ใส่ Overlay ข้อมูลบนภาพ
        cv2.putText(frame, f"LIVE | Status: {target_status} | Count: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode เป็น JPEG เพื่อส่งออก Web Stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # ส่งภาพแบบ Streaming MJPEG
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return f"""
    <html>
        <head>
            <title>AI Real-time CCTV</title>
            <style>
                body {{ background: #111; color: #0f0; text-align: center; font-family: monospace; }}
                .video-container {{ margin-top: 20px; }}
                img {{ width: 85%; border: 2px solid #333; border-radius: 8px; }}
                .info {{ font-size: 20px; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>AI MONITORING SYSTEM</h1>
            <div class="video-container">
                <img src="/video_feed">
            </div>
            <p>Streaming from: {CAMERA_PORT}</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    # บน Pi ให้เช็คว่า port 5000 ไม่ซ้ำกับระบบอื่น
    app.run(host='0.0.0.0', port=5000, threaded=True)