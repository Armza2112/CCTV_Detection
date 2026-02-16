import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from threading import Thread, Lock
from dotenv import load_dotenv
from pathlib import Path

# --- บังคับใช้ TCP เพื่อป้องกันภาพแตก/ฉีก จาก Packet Loss ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
CAMERA_PORT = int(raw_port) if raw_port.isdigit() else raw_port

# --- NCNN Setup ---
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
net = ncnn.Net()
net.opt.num_threads = 2
net.load_param(str(model_path / "model.ncnn.param"))
net.load_model(str(model_path / "model.ncnn.bin"))

app = Flask(__name__)

class AI_CCTV:
    def __init__(self, src):
        # ใช้ CAP_FFMPEG เพื่อดึงภาพแบบเสถียร
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.frame = None
        self.latest_detections = []
        self.running = True
        self.lock = Lock()
        self.status = "OFF"
        self.last_seen = 0

    def start(self):
        Thread(target=self._reader, daemon=True).start()
        Thread(target=self._inference, daemon=True).start()
        return self

    def _reader(self):
            while self.running:
                # 1. ดึงภาพออกมาทิ้งจนกว่าจะถึงเฟรมล่าสุด (สำคัญมากในการแก้ Delay)
                # เราจะไม่ยอมให้มีเฟรมค้างอยู่ใน Buffer เกิน 1 เฟรม
                while True:
                    success = self.cap.grab() # grab() เร็วกว่า read() เพราะยังไม่ต้องถอดรหัสภาพ
                    if not success: break
                    
                    # ถ้าไม่มีเฟรมเหลือในคิวแล้ว หรือกล้องส่งมาไม่ทัน ให้หยุด grab
                    if not self.cap.grab(): 
                        break

                # 2. อ่านเฟรมล่าสุดจริงๆ ออกมาถอดรหัส
                success, frame = self.cap.retrieve() 
                
                if success:
                    temp_frame = cv2.resize(frame, (640, 480))
                    with self.lock:
                        self.frame = temp_frame
                else:
                    time.sleep(0.01)

    def _inference(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue
            
            with self.lock:
                img = self.frame.copy()
            
            h, w = img.shape[:2]
            mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, 320, 320)
            mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
            
            ex = net.create_extractor()
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")
            
            detections = self.wrap_detection(mat_out, 0.4, w, h)
            with self.lock:
                self.latest_detections = detections
                if len([d for d in detections if d["class"] == 0]) > 0:
                    self.status = "ON"
                    self.last_seen = time.time()
                elif time.time() - self.last_seen > 5:
                    self.status = "OFF"

    def wrap_detection(self, mat_out, conf_threshold, img_w, img_h):
        feat = np.array(mat_out)
        if len(feat.shape) == 3: feat = feat[0].T
        elif len(feat.shape) == 2 and feat.shape[0] < feat.shape[1]: feat = feat.T
        elif len(feat.shape) == 1: 
            try: feat = feat.reshape(-1, 6)
            except: return []

        boxes, confs, ids = [], [], []
        for i in range(feat.shape[0]):
            if feat.shape[1] == 6:
                if feat[i, 1] > conf_threshold:
                    x1, y1, x2, y2 = feat[i, 2:6]
                    boxes.append([int(x1*(img_w/320)), int(y1*(img_h/320)), int((x2-x1)*(img_w/320)), int((y2-y1)*(img_h/320))])
                    confs.append(float(feat[i, 1]))
                    ids.append(int(feat[i, 0]))
            else:
                scores = feat[i, 4:]
                conf = np.max(scores)
                if conf > conf_threshold:
                    cx, cy, w, h = feat[i, 0:4]
                    boxes.append([int((cx-w/2)*(img_w/320)), int((cy-h/2)*(img_h/320)), int(w*(img_w/320)), int(h*(img_h/320))])
                    confs.append(float(conf))
                    ids.append(np.argmax(scores))

        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.45)
        return [{"box": boxes[i], "conf": confs[i], "class": ids[i]} for i in indices.flatten()] if len(indices) > 0 else []

    def get_frame_stream(self):
        desired_fps = 25
        frame_time = 1.0 / desired_fps
        
        while self.running:
            start_time = time.time()
            if self.frame is None: continue
            
            with self.lock:
                # ใช้ .copy() ภายใน Lock เพื่อป้องกันการเข้าถึง Memory ซ้อนกันจนภาพฉีก
                display_frame = self.frame.copy()
                dets = self.latest_detections
                status = self.status

            # ไม่ต้องหาร Scale เพราะเรา Resize ตั้งแต่ _reader แล้ว (แม่นยำกว่า)
            for det in dets:
                if det["class"] == 0:
                    x, y, w, h = det["box"]
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"P {det['conf']:.2f}", (x, y - 5), 0, 0.5, (0, 255, 0), 1)

            cv2.putText(display_frame, f"ST: {status}", (15, 30), 0, 0.6, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            elapsed = time.time() - start_time
            if frame_time > elapsed:
                time.sleep(frame_time - elapsed)

cctv = AI_CCTV(CAMERA_PORT).start()

@app.route('/video_feed')
def video_feed():
    return Response(cctv.get_frame_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='background:#000;text-align:center;'><img src='/video_feed' style='width:90%;'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)