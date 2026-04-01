from ultralytics import YOLO
import os

model_path = "../best.pt" 

if not os.path.exists(model_path):
    print(f"Cant find{model_path}")
else:

    model = YOLO(model_path)

    model.export(format="ncnn", imgsz=640)

    print("Success")