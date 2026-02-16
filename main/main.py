import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from threading import Thread, Lock
from dotenv import load_dotenv
from pathlib import Path

# --- Setup ---
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
RTSP_URL = raw_port # สมมติว่าเป็น rtsp://...

# --- NCNN Setup ---
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
net = ncnn.Net()
net.opt.num_threads = 4
net.load_param(str(model_path / "model.ncnn.param"))
net.load_model(str(model_path / "model.ncnn.bin"))

app = Flask(__name__)

class GStreamerCCTV:
    def __init__(self, url):
        # GStreamer Pipeline: ใช้ 'rtspsrc' และ 'omxh264dec' (หรือ v4l2h264dec) เพื่อถอดรหัสด้วย Hardware
        self.pipeline = (
            f'rtspsrc location={url} latency=100 ! '
            'rtph264depay ! h264parse ! v4l2h264dec ! '
            'videoconvert ! video/x-raw, format=BGR ! appsink drop=True'
        )
        
        # ถ้าเป็นกล้อง USB ให้ใช้ pipeline นี้แทน:
        # self.pipeline = 'v4l2src device=/dev/video0 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=True'

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.frame = None
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
        while self.running:
            success, frame = self.cap.read()
            if success:
                # Resize ทันทีเพื่อลดภาระ
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
            
            # รัน AI (320x320)
            mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 480, 320, 320)
            mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
            
            ex = net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
            
            # Post-process (Logic เดิมที่คุณมี)
            feat = np.array(mat_out)
            if len(feat.shape) == 3: feat = feat[0].T
            elif len(feat.shape) == 1: feat = feat.reshape(-1, 6)
            
            current_dets = []
            for i in range(feat.shape[0]):
                conf = feat[i, 1] if feat.shape[1] == 6 else np.max(feat[i, 4:])
                if conf > 0.4:
                    cls = int(feat[i, 0]) if feat.shape[1] == 6 else np.argmax(feat[i, 4:])
                    if cls == 0: # Person
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
            if self.frame is None: continue
            
            with self.lock:
                draw_frame = self.frame.copy()
                dets = self.latest_dets
                st = self.status

            for d in dets:
                x, y, w, h = [int(v * 2) for v in d["box"]] # Scale 320->640
                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(draw_frame, f"GStreamer HW Decode - {st}", (20, 40), 0, 0.6, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

cctv = GStreamerCCTV(RTSP_URL).start()

@app.route('/video_feed')
def video_feed():
    return Response(cctv.get_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='margin:0;background:#000;text-align:center;'><img src='/video_feed' style='height:100vh;'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)