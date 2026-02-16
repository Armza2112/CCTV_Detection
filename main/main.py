import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from threading import Thread, Lock
from dotenv import load_dotenv
from pathlib import Path

# --- บังคับใช้ TCP และล้าง Buffer ทิ้งให้เร็วที่สุด ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
RTSP_URL = int(raw_port) if raw_port.isdigit() else raw_port

# --- NCNN Setup (4 Threads สำหรับ Pi 4) ---
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
net = ncnn.Net()
net.opt.num_threads = 4
net.load_param(str(model_path / "model.ncnn.param"))
net.load_model(str(model_path / "model.ncnn.bin"))

app = Flask(__name__)

class UltraFastCCTV:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # ปรับ Buffer ให้เหลือน้อยที่สุด
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.latest_dets = []
        self.lock = Lock()
        self.running = True
        self.status = "OFF"
        self.last_seen = 0

    def start(self):
        # เธรดอ่านภาพ: ใช้เทคนิคข้ามเฟรมเพื่อลด Delay
        Thread(target=self._reader, daemon=True).start()
        # เธรด AI: รันแยกอิสระ
        Thread(target=self._inference, daemon=True).start()
        return self

    def _reader(self):
        print("🚀 Camera Reader Started (FFMPEG Mode)")
        while self.running:
            # เทคนิคการ Clear Buffer: อ่านเฟรมทิ้งถ้ามีค้างอยู่
            # เพื่อให้ได้ภาพที่สดใหม่ที่สุด (แก้ Delay)
            self.cap.grab() 
            success, frame = self.cap.retrieve()
            
            if success:
                # Resize ทันทีเป็น 640x480 เพื่อความลื่น
                temp = cv2.resize(frame, (640, 480))
                with self.lock:
                    self.frame = temp
            else:
                time.sleep(0.01)

    def _inference(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue
            
            with self.lock:
                img = self.frame.copy()
            
            # ประมวลผล AI (320x320)
            mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 480, 320, 320)
            mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
            
            ex = net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
            
            # แปลงผลลัพธ์ (Logic เดิม)
            feat = np.array(mat_out)
            if len(feat.shape) == 3: feat = feat[0].T
            elif len(feat.shape) == 1: feat = feat.reshape(-1, 6)
            
            current_dets = []
            for i in range(feat.shape[0]):
                conf = feat[i, 1] if feat.shape[1] == 6 else np.max(feat[i, 4:])
                if conf > 0.4:
                    cls = int(feat[i, 0]) if feat.shape[1] == 6 else np.argmax(feat[i, 4:])
                    if cls == 0:
                        current_dets.append({"box": feat[i, 2:6] if feat.shape[1] == 6 else feat[i, 0:4], "conf": conf})
            
            with self.lock:
                self.latest_dets = current_dets
                if len(current_dets) > 0:
                    self.status = "ON"
                    self.last_seen = time.time()
                elif time.time() - self.last_seen > 5:
                    self.status = "OFF"

    def get_stream(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.1)
                continue
            
            with self.lock:
                draw_frame = self.frame.copy()
                dets = self.latest_dets
                st = self.status

            # วาดกรอบสี่เหลี่ยม (Scale 320 -> 640)
            for d in dets:
                x, y, w, h = [int(v * 2) for v in d["box"]]
                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(draw_frame, f"MODE: FFMPEG | STATUS: {st}", (20, 40), 0, 0.6, (0, 255, 0), 2)
            
            # บีบอัดภาพ (Quality 80 คือจุดที่ชัดและลื่นที่สุด)
            _, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

cctv = UltraFastCCTV(RTSP_URL).start()

@app.route('/video_feed')
def video_feed():
    return Response(cctv.get_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='margin:0;background:#000;text-align:center;'><img src='/video_feed' style='height:100vh;'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)