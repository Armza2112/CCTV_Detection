import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

CAPTURE_DIR = BASE_DIR / "captures"
RAW_DIR = BASE_DIR / "raw_cap"
MODEL_DIR = BASE_DIR / "models"

CAPTURE_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

INTERVAL_MINUTES = 10
CONF_THRES = 0.4

load_dotenv(dotenv_path=BASE_DIR / ".env")
raw_port = os.getenv("CAMERA_PORT", "0")
RTSP_URL = int(raw_port) if raw_port.isdigit() else raw_port
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
TB_HOST = os.getenv("TB_HOST")
TB_ACCESS_TOKEN = os.getenv("TB_ACCESS_TOKEN")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"