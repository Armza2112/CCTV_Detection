import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from threading import Thread, Lock
import queue # เพิ่ม Queue เข้ามาจัดการ
from dotenv import load_dotenv
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
RTSP_URL = int(raw_port) if raw_port.isdigit() else raw_port

net = ncnn.Net()
net.opt.num_threads = 4
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
net.load_param(str(model_path / "model.ncnn.param"))
net.load_model(str(model_path / "model.ncnn.bin"))

app = Flask(__name__)

class ZeroDelayCCTV:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.q = queue.Queue(maxsize=1) # จองที่ว่างแค่ 1 เฟรมเท่านั้น!
        self.latest_dets = []
        self.lock = Lock()
        self.running = True
        self.status = "OFF"
        self.last_seen = 0

    def start(self):
        Thread(target=self._reader, daemon=True).start()
        Thread(target=self._inference, daemon=True).start()
        return self

    def _reader(self):
        print("🚀 Zero Delay Reader Started...")
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue
            
            # ถ้าใน Queue มีเฟรมเก่าค้างอยู่ ให้เอาออก (Flush) เพื่อเอาเฟรมใหม่ใส่แทน
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            # ใส่เฟรมล่าสุดลงไป
            self.q.put(cv2.resize(frame, (640, 480)))

    def _inference(self):
        while self.running:
            if self.q.empty():
                time.sleep(0.01)
                continue
            
            # ดึงภาพล่าสุดจาก Queue มาทำ AI
            img = self.q.get() 
            # ใส่กลับเข้าไปเพื่อให้เธรด Stream ดึงไปใช้ต่อได้ (หรือใช้สำเนาภาพแยก)
            self.current_frame = img.copy()

            mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 480, 320, 320)
            mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
            
            ex = net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
            
            # ... (Logic การแกะผลลัพธ์เหมือนเดิม) ...
            feat = np.array(mat_out)
            if len(feat.shape) == 3: feat = feat[0].T
            elif len(feat.shape) == 1: feat = feat.reshape(-1, 6)
            
            dets = []
            for i in range(feat.shape[0]):
                conf = feat[i, 1] if feat.shape[1] == 6 else np.max(feat[i, 4:])
                if conf > 0.4:
                    cls = int(feat[i, 0]) if feat.shape[1] == 6 else np.argmax(feat[i, 4:])
                    if cls == 0:
                        dets.append({"box": feat[i, 2:6] if feat.shape[1] == 6 else feat[i, 0:4], "conf": conf})
            
            with self.lock:
                self.latest_dets = dets
                if len(dets) > 0:
                    self.status = "ON"
                    self.last_seen = time.time()
                elif time.time() - self.last_seen > 5:
                    self.status = "OFF"

    def get_stream(self):
        while self.running:
            # ดึงภาพสดๆ จาก Queue มาโชว์
            if not hasattr(self, 'current_frame'):
                time.sleep(0.1)
                continue
            
            with self.lock:
                draw_frame = self.current_frame.copy()
                dets = self.latest_dets
                st = self.status

            for d in dets:
                x, y, w, h = [int(v * 2) for v in d["box"]]
                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(draw_frame, f"REAL-TIME | {st}", (20, 40), 0, 0.6, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

cctv = ZeroDelayCCTV(RTSP_URL).start()
# ... (Route Flask เหมือนเดิม) ...
@app.route('/video_feed')
def video_feed():
    return Response(cctv.get_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='margin:0;background:#000;text-align:center;'><img src='/video_feed' style='height:100vh;'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)