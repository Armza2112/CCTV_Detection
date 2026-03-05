import cv2
import time
import requests
import base64
import json
from config import (
    RTSP_URL, CONF_THRES, INTERVAL_MINUTES,
    RAW_DIR, CAPTURE_DIR, IMGBB_API_KEY,TB_HOST, TB_ACCESS_TOKEN
)

def upload_to_imgbb(image_path):
    try:
        with open(image_path, "rb") as file:
            payload = {
                "key": IMGBB_API_KEY,
                "image": base64.b64encode(file.read()),
            }
            response = requests.post("https://api.imgbb.com/1/upload", payload, timeout=15)
            res_data = response.json()
            
            if res_data["status"] == 200:
                image_url = res_data["data"]["url"]
                print(f"[INFO] Uploaded to ImgBB: {image_url}")
                return image_url
            else:
                print(f"[ERROR] ImgBB Upload failed: {res_data}")
                return None
    except Exception as e:
        print(f"[ERROR] Error during upload: {e}")
        return None

def snapshot_job(model, socketio, web_state):
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print("Camera not found (No route to host).")
            web_state["last_time"] = "Camera Offline (Retrying...)"
            cap.release() 
            socketio.sleep(10) 
            continue

        time.sleep(2)
        ret, frame = cap.read()
        if ret:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            raw_file = RAW_DIR / f"raw_{timestamp}.jpg"
            cv2.imwrite(str(raw_file), frame)

            results = model.predict(frame, conf=CONF_THRES, imgsz=640, verbose=False)
            annotated = results[0].plot()

            count_person = 0
            for box in results[0].boxes:
                if int(box.cls == 0):
                    count_person +=1
            
            print(count_person)

            filename = f"detect_{timestamp}.jpg"
            save_path = CAPTURE_DIR / filename
            cv2.imwrite(str(save_path), annotated)

            img_url = upload_to_imgbb(str(save_path))

            if img_url and TB_ACCESS_TOKEN:
                try:
                    tb_url = f"https://{TB_HOST}/api/v1/{TB_ACCESS_TOKEN}/telemetry"
                    response = requests.post(tb_url, json={"cctv_url": img_url, "count_person" : count_person}, timeout=5)                   
                    if response.status_code == 200:
                        print(f"[TB] Success: Link sent to ThingsBoard")
                    else:
                        print(f"[TB] Failed: Status Code {response.status_code}, Response: {response.text}")
                        
                except Exception as e:
                    print(f"[TB] Error during request: {e}")

            web_state["latest_img"] = filename
            web_state["last_time"] = time.strftime("%H:%M:%S")

            socketio.emit("new_detection", {
                "img_name": filename,
                "time": web_state["last_time"],
                "cloud_url": img_url 
            })

            print(f"Snapshot saved: {filename}")

        cap.release()
        socketio.sleep(INTERVAL_MINUTES * 60)