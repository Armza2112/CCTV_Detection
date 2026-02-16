import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from threading import Thread, Lock
from dotenv import load_dotenv
from pathlib import Path

# บังคับ TCP และปิด Buffer ของ FFMPEG ให้เหลือน้อยที่สุด
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|xerror"

base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
CAMERA_PORT = int(raw_port) if raw_port.isdigit() else raw_port

# --- NCNN Optimized ---
net = ncnn.Net()
net.opt.num_threads = 4 # เร่งกลับมาใช้ 4 core แต่เราจะคุมจังหวะรันแทน
net.opt.use_vulkan_compute = False
net.load_param(str(base_dir.parent / "train_model/best_ncnn_model/model.ncnn.param"))
net.load_model(str(base_dir.parent / "train_model/best_ncnn_model/model.ncnn.bin"))

app = Flask(__name__)

class FastAI:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.frame = None
        self.latest_dets = []
        self.status = "OFF"
        self.lock = Lock()
        self.running = True
        self.last_seen = 0

    def start(self):
        # เธรด 1: อ่านกล้องอย่างเดียว (ห้ามทำอย่างอื่น)
        Thread(target=self._read_loop, daemon=True).start()
        # เธรด 2: รัน AI อย่างเดียว (วนรันให้เร็วที่สุดเท่าที่ CPU จะไหว)
        Thread(target=self._ai_loop, daemon=True).start()
        return self

    def _read_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                # Resize ทันทีครั้งเดียวเพื่อลดภาระทุกเธรด
                temp = cv2.resize(frame, (640, 480))
                with self.lock:
                    self.frame = temp
            else:
                time.sleep(0.01)

    def _ai_loop(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue
            
            # ดึงภาพล่าสุดไปรัน AI (Copy ออกมาเพื่อไม่ให้กวนเธรดอ่าน)
            with self.lock:
                img_for_ai = self.frame.copy()
            
            # รัน AI (ใช้ขนาด 320x320 ตามโมเดล)
            mat_in = ncnn.Mat.from_pixels_resize(img_for_ai, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 480, 320, 320)
            mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
            
            ex = net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
            
            # ดึงพิกัด (wrap_detection logic เดิมของคุณ)
            dets = self.process_dets(mat_out)
            
            with self.lock:
                self.latest_dets = dets
                if len(dets) > 0:
                    self.status = "ON"
                    self.last_seen = time.time()
                elif time.time() - self.last_seen > 5:
                    self.status = "OFF"
            
            # พักจังหวะเล็กน้อยไม่ให้ CPU ร้อนเกิน (ปรับเพิ่มได้ถ้าเครื่องค้าง)
            time.sleep(0.01)

    def process_dets(self, mat_out):
        feat = np.array(mat_out)
        if len(feat.shape) == 3: feat = feat[0].T
        elif len(feat.shape) == 1:
            try: feat = feat.reshape(-1, 6)
            except: return []
        
        results = []
        conf_threshold = 0.4
        for i in range(feat.shape[0]):
            conf = feat[i, 1] if feat.shape[1] == 6 else np.max(feat[i, 4:])
            if conf > conf_threshold:
                cls = int(feat[i, 0]) if feat.shape[1] == 6 else np.argmax(feat[i, 4:])
                if cls == 0: # Person
                    results.append({"box": feat[i, 2:6] if feat.shape[1] == 6 else feat[i, 0:4], "conf": conf})
        return results

    def stream(self):
        while self.running:
            if self.frame is None: continue
            
            with self.lock:
                draw_frame = self.frame.copy()
                dets = self.latest_dets
                st = self.status

            # วาดกรอบ (ใช้พิกัดล่าสุดที่ AI หาเจอ)
            for d in dets:
                # พิกัดจากโมเดล 320 -> 640 (คูณ 2)
                # หมายเหตุ: ปรับสูตรตามโมเดลของคุณ ถ้ากรอบไม่ตรง
                x, y, w, h = [int(v * 2) for v in d["box"]]
                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(draw_frame, f"LIVE - {st}", (20, 40), 0, 0.7, (0, 255, 0), 2)
            
            # บีบอัดแบบ Turbo (Quality 75)
            _, buf = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            
            # ไม่ต้อง sleep ในเธรดส่ง เพื่อให้ภาพไหลตามความเร็ว CPU/Network ทันที

cctv = FastAI(CAMERA_PORT).start()

@app.route('/video_feed')
def video_feed():
    return Response(cctv.stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='margin:0;background:#000;display:flex;justify-content:center;'><img src='/video_feed' style='height:100vh;'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)