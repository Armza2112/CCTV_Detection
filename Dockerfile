FROM python:3.10-slim

# 1. ติดตั้ง Library พื้นฐานสำหรับ OpenCV และการคำนวณของ AI
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# 2. ติดตั้ง Torch สำหรับ ARM64 จากแหล่งที่ถูกต้อง (ป้องกัน Illegal Instruction)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. ติดตั้ง Ultralytics โดยไม่ให้ไปลง Torch ทับตัวที่เราลงไว้
RUN pip install --no-cache-dir ultralytics --no-deps

# 4. ติดตั้ง Library อื่นๆ จาก requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. คัดลอกโค้ดทั้งหมด (รวมถึงโฟลเดอร์ models ที่มี best.pt)
COPY . .

CMD [ "python", "main.py" ]