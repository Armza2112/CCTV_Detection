import cv2
import time
import os
import numpy as np
from flask import Flask, Response
import ncnn  
from dotenv import load_dotenv
from pathlib import Path

# --- Setup ---
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
CAMERA_PORT = os.getenv("CAMERA_PORT", "0")

# เช็คว่าถ้าเป็นตัวเลขล้วน (เช่น "0" หรือ "1") ให้แปลงเป็น int
# แต่ถ้าเป็นตัวหนังสือ (เช่น "rtsp://...") ให้ปล่อยเป็น string ไว้เหมือนเดิม
if CAMERA_PORT.isdigit():
    CAMERA_PORT = int(CAMERA_PORT)

# --- NCNN Setup ---
model_path = base_dir.parent / "train_model" / "best_ncnn_model"
net = ncnn.Net()
net.opt.num_threads = 4
net.load_param(str(model_path / "model.ncnn.param"))
net.load_model(str(model_path / "model.ncnn.bin"))

app = Flask(__name__)
person_count = 0
target_status = "OFF"
last_seen_time = 0
off_delay = 5

def wrap_detection(mat_out, conf_threshold, img_w, img_h):
    # แปลงเป็น numpy array
    feat = np.array(mat_out)
    
    # แก้ปัญหา IndexError: เช็คว่าถ้ามาเป็น 1 มิติ ให้พยายาม Reshape กลับ
    # ปกติ YOLOv8 NCNN จะมีพารามิเตอร์ประมาณ 84 หรือ 64
    if len(feat.shape) == 1:
        # ถ้ามาเป็นเส้นตรง ให้ลองเปลี่ยนเป็น [n, 6] (รูปแบบ DetectionOutput มาตรฐาน)
        # หรือถ้ามันมั่ว ให้ลองเปลี่ยนเป็น feat = feat.reshape(-1, 84) 
        try:
            feat = feat.reshape(-1, 6) # NCNN ส่วนใหญ่ที่ Export สำเร็จรูปจะใช้ format นี้
        except:
            # ถ้า reshape ไม่ได้ แสดงว่าเป็นรูปแบบอื่น ให้ส่งค่าว่างไปก่อนเพื่อไม่ให้โปรแกรมค้าง
            return []

    # ถ้ามาเป็น [1, 84, 2100] ให้บีบเหลือ [84, 2100] และ Transpose
    if len(feat.shape) == 3:
        feat = feat[0].T
    elif len(feat.shape) == 2 and feat.shape[0] < feat.shape[1]:
        feat = feat.T

    boxes = []
    confidences = []
    class_ids = []
    
    for i in range(feat.shape[0]):
        # ตรวจสอบรูปแบบข้อมูล: [class_id, conf, x1, y1, x2, y2] หรือ [x, y, w, h, scores...]
        if feat.shape[1] == 6:  # แบบสำเร็จรูป (DetectionOutput)
            conf = feat[i, 1]
            if conf > conf_threshold:
                class_id = int(feat[i, 0])
                # พิกัดมักเป็นค่า 0-1 หรือพิกเซล 320x320
                x1, y1, x2, y2 = feat[i, 2:6]
                boxes.append([int(x1 * (img_w/320)), int(y1 * (img_h/320)), 
                              int((x2-x1) * (img_w/320)), int((y2-y1) * (img_h/320))])
                confidences.append(float(conf))
                class_ids.append(class_id)
        else:  # แบบ Raw [x, y, w, h, scores...]
            scores = feat[i, 4:]
            conf = np.max(scores)
            if conf > conf_threshold:
                class_id = np.argmax(scores)
                cx, cy, w, h = feat[i, 0:4]
                x = int((cx - w/2) * (img_w / 320))
                y = int((cy - h/2) * (img_h / 320))
                boxes.append([x, y, int(w * (img_w/320)), int(h * (img_h/320))])
                confidences.append(float(conf))
                class_ids.append(class_id)
            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
    
    final_results = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_results.append({
                "box": boxes[i],
                "conf": confidences[i],
                "class": class_ids[i]
            })
    return final_results
def generate_frames():
    global person_count, target_status, last_seen_time
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success: break
        
        h, w = frame.shape[:2]
        
        # 1. Preprocess: Resize & Normalize
        mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, 320, 320)
        mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
        
        # 2. Inference
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        
        # 3. Post-process (Decoding + NMS)
        detections = wrap_detection(mat_out, 0.4, w, h)
        
        # 4. Draw & Logic
        current_count = 0
        for det in detections:
            # สมมติว่าต้องการตรวจจับคน (Class 0)
            if det["class"] == 0:
                current_count += 1
                x, y, bw, bh = det["box"]
                # วาดกรอบ
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {det['conf']:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Logic ON/OFF
        person_count = current_count
        current_time = time.time()
        if person_count > 0:
            last_seen_time = current_time
            target_status = "ON"
        else:
            target_status = "OFF" if (current_time - last_seen_time > off_delay) else "ON"

        # Overlay ข้อมูล
        cv2.putText(frame, f"STATUS: {target_status} | COUNT: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return f"<html><body style='background:#111;color:#0f0;text-align:center;'><h1>CCTV MONITORING</h1><img src='/video_feed' style='width:80%;'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)