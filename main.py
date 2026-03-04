import cv2
import time
import os
from ultralytics import YOLO
from pathlib import Path
from flask import Flask, send_from_directory, render_template_string
from flask_socketio import SocketIO, emit
from threading import Thread
from dotenv import load_dotenv

# ================= CONFIG =================
INTERVAL_MINUTES = 10 
CONF_THRES = 0.4
base_dir = Path(__file__).resolve().parent
save_dir = base_dir / "captures"
raw_dir = base_dir / "raw_cap"
raw_dir.mkdir(exist_ok=True)
save_dir.mkdir(exist_ok=True)

load_dotenv(dotenv_path=base_dir / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
RTSP_URL = int(raw_port) if raw_port.isdigit() else raw_port
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ================= YOLO NCNN =================
model_path = base_dir / "models" / "best.pt" 

# 2. โหลดโมเดล และระบุ device="cpu" ให้ชัดเจน
model = YOLO(str(model_path))
if not (base_dir / "models" / "best_ncnn_model").exists():
    print("Exporting model to NCNN for the first time on this device...")
    model.export(format="ncnn")
model = YOLO(str(base_dir / "models" / "best_ncnn_model"), task="detect")
# ================= FLASK & SOCKET.IO =================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

latest_img_name = ""
last_update_time = "Waiting..."

@app.route('/')
def index():
    return render_template_string("""
    <html>
        <head>
            <title>AI CCTV WebSocket</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <style>
                body { background: #121212; color: #e0e0e0; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
                .card { background: #1e1e1e; max-width: 900px; margin: auto; padding: 20px; border-radius: 15px; border: 1px solid #333; }
                img { width: 100%; border-radius: 10px; border: 3px solid #4CAF50; margin: 15px 0; transition: 0.5s; }
                .status-bar { display: flex; justify-content: space-around; background: #333; padding: 12px; border-radius: 8px; margin-bottom: 10px; }
                .time { color: #4CAF50; font-weight: bold; }
                .live-tag { color: #ff4444; font-weight: bold; animation: blink 1s infinite; }
                @keyframes blink { 50% { opacity: 0; } }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>AI Smart Snapshot <span class="live-tag">● WS LIVE</span></h1>
                <div class="status-bar">
                    <div>Updata: <span id="last_time" class="time">{{ last_time }}</span></div>
                    <div>{{ interval }} min</div>
                </div>
                <img id="cctv_img" src="{{ '/download/' + img_name if img_name else '' }}" 
                     style="{{ 'display:block' if img_name else 'display:none' }}">
                <div id="wait_msg" style="{{ 'display:none' if img_name else 'display:block; padding:100px;' }}">
                    AI...
                </div>
            </div>

            <script>
                var socket = io();
                socket.on('new_detection', function(data) {
                    var img = document.getElementById('cctv_img');
                    var msg = document.getElementById('wait_msg');
                    
                    img.src = "/download/" + data.img_name + "?t=" + new Date().getTime();
                    img.style.display = "block";
                    msg.style.display = "none";
                    
                    document.getElementById('last_time').innerHTML = data.time;
                });
            </script>
        </body>
    </html>
    """, img_name=latest_img_name, last_time=last_update_time, interval=INTERVAL_MINUTES)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(save_dir, filename)

# ================= BACKGROUND SNAPSHOT JOB =================
def snapshot_job():
    global latest_img_name, last_update_time
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            time.sleep(2)
            ret, frame = cap.read()
            if ret:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                raw_filename = f"raw_{timestamp}.jpg"
                cv2.imwrite(str(raw_dir / raw_filename), frame)
                results = model.predict(frame, conf=CONF_THRES, imgsz=640, verbose=False)
                annotated_frame = results[0].plot()
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"detect_{timestamp}.jpg"
                cv2.imwrite(str(save_dir / filename), annotated_frame)
                
                latest_img_name = filename
                last_update_time = time.strftime("%H:%M:%S")
                
                socketio.emit('new_detection', {
                    'img_name': filename,
                    'time': last_update_time
                })
            
            cap.release()
        
        socketio.sleep(INTERVAL_MINUTES * 60)
# ================= RUN =================
if __name__ == "__main__":
    socketio.start_background_task(snapshot_job)
    
    print("Web Server running on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000)