import cv2
import time
import requests
import base64
import json
import paho.mqtt.client as mqtt
import numpy as np
from config import (
    RTSP_URL, CONF_THRES, INTERVAL_MINUTES,
    CAPTURE_DIR, IMGBB_API_KEY,TB_HOST, TB_ACCESS_TOKEN, MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
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
                print(f"Uploaded to ImgBB: {image_url}")
                return image_url
            else:
                print(f"[ERROR] ImgBB Upload failed: {res_data}")
                return None
    except Exception as e:
        print(f"[ERROR] Error during upload: {e}")
        return None
    
def send_mqtt(switch, state):
    try:
        client = mqtt.Client()
        client.connect(MQTT_BROKER, int(MQTT_PORT), 60)
        
        payload = {switch: state}
        
        client.publish(MQTT_TOPIC, json.dumps(payload))
        
        client.disconnect()
        print(f"[MQTT] Sent success {switch} : {state}")
    except Exception as e:
        print(f"[MQTT] Error: {e}")

def snapshot_job(model, socketio, web_state):
    last_mqtt_state = None
    prev_frame = None
    
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            web_state["last_time"] = "Camera Offline"
            socketio.sleep(10) 
            continue

        time.sleep(2)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            motion_detected = False
            is_first_run = (prev_frame is None)

            if not is_first_run:
                frame_delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                cv2.imwrite(str(CAPTURE_DIR / "motion_debug.jpg"), thresh)
                
                socketio.emit("update_motion", {"time": time.strftime("%H:%M:%S")})

                motion_score = np.sum(thresh)
                if motion_score > 5000: 
                    motion_detected = True
                    print(f"[MOTION] Detected! Score: {motion_score}")

            prev_frame = gray

            if motion_detected or is_first_run:
                # Predict
                results = model.predict(frame, conf=CONF_THRES, imgsz=640, verbose=False)
                annotated = results[0].plot()

                count_person = sum(1 for box in results[0].boxes if int(box.cls) == 0)
                
                filename = "latest_detect.jpg" 
                save_path = CAPTURE_DIR / filename
                cv2.imwrite(str(save_path), annotated)

                img_url = upload_to_imgbb(str(save_path))

                # ThingsBoard & MQTT 
                if img_url and TB_ACCESS_TOKEN:
                    try:
                        tb_url = f"https://{TB_HOST}/api/v1/{TB_ACCESS_TOKEN}/telemetry"
                        requests.post(tb_url, json={"cctv_url": img_url, "count_person": count_person}, timeout=5)
                    except: pass

                current_state = "ON" if count_person > 0 else "OFF"
                if MQTT_BROKER and current_state != last_mqtt_state:
                    send_mqtt("state_left", current_state)
                    last_mqtt_state = current_state
                
                web_state["latest_img"] = filename
                web_state["last_time"] = time.strftime("%H:%M:%S")
                socketio.emit("new_detection", {
                    "img_name": filename,
                    "time": web_state["last_time"],
                    "cloud_url": img_url 
                })
                print(f"Preson:{count_person}")
            else:
                print("No motion detected.")

        socketio.sleep(INTERVAL_MINUTES * 60)