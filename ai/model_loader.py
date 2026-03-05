from ultralytics import YOLO
from config import MODEL_DIR

def load_model():
    pt_path = MODEL_DIR / "best.pt"
    ncnn_path = MODEL_DIR / "best_ncnn_model"

    if ncnn_path.exists():
        print("Found NCNN model. Loading High Performance Mode...")
        return YOLO(str(ncnn_path), task="detect")

    elif pt_path.exists():
        print(f"NCNN model not found. Converting from {pt_path.name}...")
        try:
            tmp_model = YOLO(str(pt_path))
            tmp_model.export(format="ncnn", imgsz=640)
            print("Conversion successful!")
            return YOLO(str(ncnn_path), task="detect")
        except Exception as e:
            print(f"Conversion failed: {e}")
            print("Falling back to PyTorch (.pt) mode...")
            return YOLO(str(pt_path))
    else:
        print("Error: No model found")
        return None