"""Export the trained PyTorch gauge model to OpenVINO.

Usage:
    python scripts/export_openvino.py
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "app" / "services" / "gauge_detection" / "models" / "gauge.pt"
OUTPUT_DIR = ROOT / "app" / "services" / "gauge_detection" / "models"

if not MODEL.exists():
    raise SystemExit(f"Missing model weights: {MODEL}")

model = YOLO(str(MODEL), task="detect")
exported = model.export(format="openvino", imgsz=640, dynamic=False, device="cpu")
print("Exported model:", exported)
print("The detector will automatically prefer the OpenVINO export when present.")