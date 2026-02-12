import cv2
import os
import time
from datetime import datetime
from pathlib import Path

# --- ตั้งค่าใหม่ ---
# ใช้ Path(__file__) เพื่อหาโฟลเดอร์ที่รันโค้ดอยู่ปัจจุบัน และสร้างโฟลเดอร์ image ไว้ข้างใน
base_dir = Path(__file__).resolve().parent
save_folder = os.path.join(base_dir, "image")

rtsp_url = "rtsp://admin:DNJYCE@192.168.11.44:554/Streaming/channels/101/" 
interval = 300  # 10 นาที = 600 วินาที

# สร้างโฟลเดอร์ image ถ้ายังไม่มี
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"📁 สร้างโฟลเดอร์เก็บภาพที่: {save_folder}")

def capture():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_folder, f"img_{timestamp}.jpg")

    # แนะนำ: สำหรับ RTSP บางครั้งต้องเปิดทิ้งไว้สักครู่เพื่อให้ภาพหายเบลอ
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"[{timestamp}] ❌ Error: Connect กล้องไม่ได้")
        return

    # อ่านเฟรม (อาจจะอ่านทิ้งสัก 2-3 เฟรมเพื่อให้ได้ภาพที่นิ่ง)
    for _ in range(5):
        cap.read()
    
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(file_path, frame)
        print(f"[{timestamp}] ✅ บันทึกภาพเรียบร้อย: {file_path}")
    else:
        print(f"[{timestamp}] ❌ Error: อ่านเฟรมภาพไม่ได้")
    
    cap.release()

if __name__ == "__main__":
    print(f"🚀 เริ่มระบบแคปภาพอัตโนมัติ (ทุก {interval/60:.0f} นาที)")
    print(f"📍 บันทึกไปที่: {save_folder}")
    
    try:
        while True:
            capture()
            print(f"💤 กำลังรอรอบถัดไปในอีก {interval/60:.0f} นาที...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 หยุดการทำงานของระบบ")