FROM python:3.10-slim-bullseye

WORKDIR /app

# install lib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "numpy<2.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "numpy<2.0" --force-reinstall

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]