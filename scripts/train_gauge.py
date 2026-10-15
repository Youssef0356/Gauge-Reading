"""
Overnight gauge model training script.

Uses a dedicated virtual environment with CUDA PyTorch so it stays
independent of the production backend env (which is CPU-only on purpose).

Usage
-----
1. One-time setup (creates the venv + installs deps):
       python scripts/train_gauge.py --setup

2. Start training overnight:
       python scripts/train_gauge.py --train

3. Resume a previously interrupted run:
       python scripts/train_gauge.py --train --resume

The trained weights land in:
    scripts/gauge_training/runs/detect/<run_name>/weights/best.pt

Copy best.pt to:
    app/services/computer_vision/models/gauge.pt
Then re-export to OpenVINO:
    uv run model-convert --model gauge --target openvino
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR / "gauge_training"
VENV_DIR = TRAINING_DIR / ".venv"
DATASET_DIR = TRAINING_DIR / "dataset"
RUNS_DIR = TRAINING_DIR / "runs"

# Your existing production model — used as the starting checkpoint so the
# fine-tuned model benefits from weights already learned on gauge images.
PRODUCTION_MODEL = (
    SCRIPT_DIR.parent
    / "app"
    / "services"
    / "computer_vision"
    / "models"
    / "gauge.pt"
)

# ---------------------------------------------------------------------------
# Roboflow dataset
# ---------------------------------------------------------------------------
# The Roboflow API key is a secret and must NOT be committed. Export it before
# running (PowerShell:  $env:ROBOFLOW_API_KEY = "..."  / bash: export ROBOFLOW_API_KEY="...").
ROBOFLOW_WORKSPACE = "projectsolardefectsclassification"
ROBOFLOW_PROJECT = "gauge-analog-vff2w"
ROBOFLOW_VERSION = 3

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_CONFIG = dict(
    epochs=300,          # max 300 — early stopping will kick in automatically
    imgsz=640,
    batch=8,             # safe for RTX 4060 8 GB VRAM
    dropout=True,
    patience=50,         # auto-stop if no improvement for 50 consecutive epochs
    device=0,            # GPU 0 (RTX 4060)
    workers=4,
    project=str(RUNS_DIR),
    name="gauge_finetune",
    exist_ok=False,      # set True when resuming
    save=True,
    plots=True,
    amp=True,            # automatic mixed precision — faster on RTX
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python() -> Path:
    """Path to the venv Python executable."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup() -> None:
    """Create a CUDA-capable venv and install ultralytics + roboflow."""
    print("=" * 60)
    print("Setting up gauge training environment")
    print("=" * 60)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

    py = _python()

    # Upgrade pip quietly
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # Install CUDA PyTorch first (cu124 works with driver 617+)
    run([
        str(py), "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "-q",
    ])

    # Install ultralytics and roboflow
    run([str(py), "-m", "pip", "install", "ultralytics", "roboflow", "-q"])

    print("\n Setup complete. Run with --train to start training.\n")


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download the Roboflow dataset and return path to data.yaml."""
    # Check if we already have a resolved location from a previous download
    marker_file = TRAINING_DIR / "dataset_location.txt"
    if marker_file.exists():
        yaml_path = Path(marker_file.read_text(encoding="utf-8").strip())
        if yaml_path.exists():
            print(f"Dataset already at: {yaml_path}. Skipping download.")
            return yaml_path

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # Download via roboflow inside the training venv.
    # IMPORTANT: do NOT pre-create DATASET_DIR — if the folder already exists,
    # Roboflow silently skips the download. Let it create the folder itself.
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Export it before running "
            "(PowerShell: $env:ROBOFLOW_API_KEY = '...')."
        )

    location_file = TRAINING_DIR / "_rf_location.txt"
    # The key is passed through the child process environment, never embedded
    # in the -c script text (that would expose it in the process command line).
    download_script = f"""
import os
import sys
from pathlib import Path
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("{ROBOFLOW_WORKSPACE}").project("{ROBOFLOW_PROJECT}")
version = project.version({ROBOFLOW_VERSION})
dataset = version.download("yolov12", location=r"{DATASET_DIR}")
loc = dataset.location
print("Dataset location:", loc)
Path(r"{location_file}").write_text(loc, encoding="utf-8")
"""
    py = _python()
    run([str(py), "-c", download_script], env={**os.environ, "ROBOFLOW_API_KEY": api_key})

    if not location_file.exists():
        raise RuntimeError("Roboflow download did not write dataset location.")

    rf_location = Path(location_file.read_text(encoding="utf-8").strip())
    print(f"Roboflow dataset root: {rf_location}")

    # Find data.yaml — may be directly in rf_location or one level deeper
    yaml_path = rf_location / "data.yaml"
    if not yaml_path.exists():
        candidates = list(rf_location.rglob("data.yaml"))
        if not candidates:
            raise FileNotFoundError(
                f"data.yaml not found anywhere under {rf_location}"
            )
        yaml_path = candidates[0]
    dataset_root = yaml_path.parent

    # Rewrite train/val/test paths to absolute so YOLO works from any cwd
    content = yaml_path.read_text(encoding="utf-8")
    lines = [
        line for line in content.splitlines()
        if not line.startswith(("train:", "val:", "test:"))
    ]
    lines += [
        f"train: {dataset_root / 'train' / 'images'}",
        f"val: {dataset_root / 'valid' / 'images'}",
        f"test: {dataset_root / 'test' / 'images'}",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated data.yaml with absolute paths -> {yaml_path}")

    # Save resolved path for future runs
    marker_file.write_text(str(yaml_path), encoding="utf-8")
    return yaml_path



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(resume: bool = False) -> None:
    """Download dataset (if needed) and start training."""
    if not _python().exists():
        print("ERROR: Training venv not found. Run --setup first.")
        sys.exit(1)

    data_yaml = download_dataset()

    # Fine-tune from existing production gauge.pt for faster convergence
    if PRODUCTION_MODEL.exists():
        model_path = str(PRODUCTION_MODEL)
        print(f"Fine-tuning from production model: {model_path}")
    else:
        model_path = "yolov8m.pt"
        print(f"Production model not found, using {model_path} (will be downloaded)")

    cfg = dict(TRAIN_CONFIG)
    cfg["data"] = str(data_yaml)

    if resume:
        last_pts = sorted(RUNS_DIR.rglob("last.pt"))
        if not last_pts:
            print("ERROR: No previous run found to resume.")
            sys.exit(1)
        last_pt = last_pts[-1]
        print(f"Resuming from: {last_pt}")
        train_script = f"""
from ultralytics import YOLO
model = YOLO(r"{last_pt}")
results = model.train(resume=True)
print("Training complete. Best weights:", results.save_dir)
"""
    else:
        cfg_repr = repr(cfg)
        production_model_str = str(PRODUCTION_MODEL)
        train_script = f"""
from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(r"{model_path}")
cfg = {cfg_repr}
results = model.train(**cfg)
print("\\n=== Training complete ===")
best = results.save_dir / "weights" / "best.pt"
print("Best weights:", best)
print("\\nTo deploy, copy best.pt to:")
print(r"  {production_model_str}")
"""

    py = _python()
    run([str(py), "-c", train_script])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gauge YOLO model training - local RTX 4060"
    )
    parser.add_argument("--setup", action="store_true", help="Create venv and install deps")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.train:
        train(resume=args.resume)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
