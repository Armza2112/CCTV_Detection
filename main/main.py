import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from threading import Thread, Lock
from dotenv import load_dotenv
from pathlib import Path

# --- Setup & Config ---
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")

# จัดการ CAMERA_PORT ให้รองรับทั้ง USB และ RTSP
raw_port = os.getenv("CAMERA_PORT", "0")
CAMERA_PORT = int(raw_port) if raw_port.isdigit() else raw_port

# --- Optimized NCNN Setup ---
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
net = ncnn.Net()
net.opt.num_threads = 4  # ใช้ครบทุก Core ของ Pi 4
net.opt.use_fp16_packed = True
net.opt.use_fp16_storage = True
net.opt.use_fp16_arithmetic = True

net.load_param(str(model_path / "model.ncnn.param"))
net.load_model(str(model_path / "model.ncnn.bin"))

app = Flask(__name__)

# --- กล้องแบบ Threaded (แก้ปัญหาภาพหน่วงสะสม) ---
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # จองบัฟเฟอร์ให้น้อยที่สุดเพื่อความสดของภาพ
        self.success, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Lock() # ป้องกันการอ่านเขียนเฟรมพร้อมกัน

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01) # พักเครื่องถ้าอ่านไม่สำเร็จ

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# Global variables สำหรับสถานะระบบ
person_count = 0
target_status = "OFF"
last_seen_time = 0
off_delay = 5

def wrap_detection(mat_out, conf_threshold, img_w, img_h):
    feat = np.array(mat_out)
    if len(feat.shape) == 1:
        try: feat = feat.reshape(-1, 6)
        except: return []
    if len(feat.shape) == 3: feat = feat[0].T
    elif len(feat.shape) == 2 and feat.shape[0] < feat.shape[1]: feat = feat.T

    boxes, confidences, class_ids = [], [], []
    for i in range(feat.shape[0]):
        if feat.shape[1] == 6:
            conf = feat[i, 1]
            if conf > conf_threshold:
                x1, y1, x2, y2 = feat[i, 2:6]
                boxes.append([int(x1 * (img_w/320)), int(y1 * (img_h/320)), 
                              int((x2-x1) * (img_w/320)), int((y2-y1) * (img_h/320))])
                confidences.append(float(conf))
                class_ids.append(int(feat[i, 0]))
        else:
            scores = feat[i, 4:]
            conf = np.max(scores)
            if conf > conf_threshold:
                cx, cy, w, h = feat[i, 0:4]
                boxes.append([int((cx - w/2) * (img_w / 320)), int((cy - h/2) * (img_h / 320)), 
                              int(w * (img_w/320)), int(h * (img_h/320))])
                confidences.append(float(conf))
                class_ids.append(np.argmax(scores))
            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
    return [{"box": boxes[i], "conf": confidences[i], "class": class_ids[i]} for i in indices.flatten()] if len(indices) > 0 else []

def generate_frames():
    global person_count, target_status, last_seen_time
    
    vs = VideoStream(CAMERA_PORT).start()
    frame_count = 0
    current_detections = [] # แก้ UnboundLocalError
    current_count = 0

    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # รัน AI ทุกๆ 3 เฟรม เพื่อ FPS ที่ลื่นไหลบน Pi 4
        # (ปรับตัวเลข % เป็น 2 หรือ 1 ได้ถ้าอยากให้แม่นขึ้นแต่จะหน่วงลง)
        if frame_count % 3 == 0:
            mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, 320, 320)
            mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
            
            ex = net.create_extractor()
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")
            
            current_detections = wrap_detection(mat_out, 0.4, w, h)
            current_count = len([d for d in current_detections if d["class"] == 0])

        # วาด Bounding Boxes จากผลลัพธ์ล่าสุด
        for det in current_detections:
            if det["class"] == 0: # ตรวจจับคน
                x, y, bw, bh = det["box"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"P {det['conf']:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Logic ON/OFF
        person_count = current_count
        current_time = time.time()
        if person_count > 0:
            last_seen_time = current_time
            target_status = "ON"
        else:
            target_status = "OFF" if (current_time - last_seen_time > off_delay) else "ON"

        # ใส่ Overlay ข้อมูลหน้าจอ
        cv2.putText(frame, f"ST: {target_status} | CNT: {person_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode และส่งภาพ (ลดคุณภาพ JPEG เหลือ 70% เพื่อความลื่นไหลผ่าน Network)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <head><title>Raspberry Pi AI CCTV</title></head>
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
            <h1>AI MONITORING SYSTEM</h1>
            <img src='/video_feed' style='width:85%; border:2px solid #333;'>
        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)