import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  # เปลี่ยนจาก ultralytics เป็น ncnn
from dotenv import load_dotenv
from pathlib import Path

# --- Setup & Config ---
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")

# ดึงค่ากล้องจาก .env
CAMERA_PORT = os.getenv("CAMERA_PORT", 0)
if str(CAMERA_PORT).isdigit(): 
    CAMERA_PORT = int(CAMERA_PORT)

# --- NCNN Model Setup ---
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
param_path = str(model_path / "model.ncnn.param")
bin_path = str(model_path / "model.ncnn.bin")

# โหลดโมเดลด้วย ncnn
net = ncnn.Net()
net.load_param(param_path)
net.load_model(bin_path)

app = Flask(__name__)

# ตัวแปร Global สำหรับสถานะ
person_count = 0
target_status = "OFF"
last_seen_time = 0
off_delay = 5 

def generate_frames():
    global person_count, target_status, last_seen_time
    
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ดึงชื่อ Input/Output Layer (ปกติ YOLO NCNN จะใช้ชื่อนี้)
    input_name = "in0"
    output_name = "out0"

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # --- Inference ด้วย NCNN ---
        # 1. เตรียมภาพ (Resize เป็น 320x320 ตามที่คุณต้องการความลื่นไหล)
        img_h, img_w = frame.shape[:2]
        mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                                            img_w, img_h, 320, 320)
        
        # 2. รันโมเดล
        ex = net.create_extractor()
        ex.input(input_name, mat_in)
        ret, mat_out = ex.extract(output_name)
        
        # 3. จัดการผลลัพธ์ (NCNN output จะเป็น array ของการตรวจจับ)
        # หมายเหตุ: ncnn output ของ YOLO มักจะเป็น [object_count, 6] 
        # (class, conf, x, y, w, h)
        results = np.array(mat_out)
        current_count = 0
        
        if len(results.shape) >= 2:
            for detection in results:
                conf = detection[1]
                if conf > 0.4: # ค่า conf เดียวกับโค้ดเดิม
                    current_count += 1
                    # วาดกรอบสี่เหลี่ยม (แทน r.plot())
                    x1, y1, x2, y2 = detection[2], detection[3], detection[4], detection[5]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # --- Logic เดิมของคุณทั้งหมด ---
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
    app.run(host='0.0.0.0', port=5000, threaded=True)