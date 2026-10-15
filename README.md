# Gauge Reading Service

A small, public-ready service for reading **analog gauges** from a photo or a
single video frame. It detects the gauge `center`, `min`, `max`, `gauge`, and
`tip` points with YOLO, then converts the needle angle to a measurement.

The project includes:

- a FastAPI backend (`app/`),
- a browser-based tester (`/ui/`),
- a training workflow for building your own model,
- an optional OpenVINO export for faster CPU inference.

No proprietary dependencies are required at runtime except the trained model
weights.

---

## 1. Requirements

- Python 3.10 or newer
- A webcam/phone photos of your gauges for training or testing
- Optional but recommended for training: an NVIDIA GPU

---

## 2. Quick start (run the API and web UI)

```bash
git clone <your-repository-url>
cd "Gauge Reading"

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt

# Provide the model weights (see section 4), then:
python -m uvicorn app.main:app --reload
```

Open:

- Web tester: http://127.0.0.1:8000/ui/
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/

API endpoint:

```text
POST /api/v1/gauge/detect
```

Form fields:

- `image` (required): image file (JPEG/PNG)
- `min_value` (optional, default `0`): gauge minimum reading
- `max_value` (optional, default `100`): gauge maximum reading
- `unit` (optional, default `bar`): measurement unit
- `clockwise` (optional, default `true`): whether the needle moves clockwise
  from `min` to `max`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/gauge/detect?min_value=0&max_value=10&unit=bar&clockwise=true" \
  -F "image=@sample.jpg"
```

Successful response:

```json
{
  "status": "success",
  "value": 2.256,
  "confidence": 0.701,
  "unit": "bar",
  "fraction": 0.225628,
  "angles": {
    "min": 150.75,
    "max": 59.51,
    "tip": 211.39
  }
}
```

If the detector cannot find `center`, `min`, `max`, or `tip`, the API returns
`422` with the missing point names.

---

## 3. Get the dataset and train your own model

This repository does not ship model weights or datasets. You can train your own
model using the included script and a public Roboflow dataset.

### 3.1 Create a Roboflow account and get an API key

1. Go to https://roboflow.com and create a free account.
2. Open your workspace settings:
   https://app.roboflow.com/settings/api
3. Copy your **Private API Key**.
4. Create a local `.env` file:

   ```bash
   copy .env.example .env
   ```

   Then edit `.env`:

   ```env
   ROBOFLOW_API_KEY=your_actual_roboflow_api_key
   ```

The training script reads `ROBOFLOW_API_KEY` from the environment. **Never**
commit `.env`.

### 3.2 Download the dataset and train

```bash
python scripts/train_gauge.py --setup
python scripts/train_gauge.py --train
```

What happens:

- a training-only Python environment is created in
  `scripts/gauge_training/.venv`,
- the public `gauge-analog` dataset is downloaded from Roboflow,
- YOLO trains on your machine (GPU if available, otherwise CPU).

Training output is written to:

```text
scripts/gauge_training/runs/detect/gauge_finetune/weights/best.pt
```

### 3.3 Install the trained weights

Copy the best weights into the model folder:

```bash
copy scripts\gauge_training\runs\detect\gauge_finetune\weights\best.pt app\services\gauge_detection\models\gauge.pt
```

On macOS/Linux:

```bash
cp scripts/gauge_training/runs/detect/gauge_finetune/weights/best.pt app/services/gauge_detection/models/gauge.pt
```

### 3.4 Optional: export to OpenVINO for faster CPU inference

```bash
python scripts/export_openvino.py
```

The detector automatically uses the OpenVINO export
(`gauge_openvino_model/`) when it exists and otherwise falls back to
`gauge.pt`.

---

## 4. Model files expected by the service

The detector looks in:

```text
app/services/gauge_detection/models/
```

Preferred (fastest on CPU):

```text
app/services/gauge_detection/models/gauge_openvino_model/
  gauge.xml
  gauge.bin
  metadata.yaml
```

Fallback:

```text
app/services/gauge_detection/models/gauge.pt
```

Model weights are intentionally not committed to Git because they are large.
See the publishing notes below.

---

## 5. Published repository notes

This project is intended to be public. Before pushing, ensure:

- `.env` is not committed.
- No API keys, passwords, or private URLs are in the source.
- Training datasets, run folders, caches, and model weights are ignored.
- Large files are handled with Git LFS or GitHub Releases if you want to
  distribute them.

Recommended first-time setup:

```bash
git init
git add .
git status
git commit -m "Initial public release"
```

If Git asks to track model weights, do **not** commit them. Instead attach them
to a GitHub Release or use Git LFS.

### Rotate leaked credentials

If an API key was ever committed, even in an earlier commit:

1. Revoke/regenerate the key in Roboflow.
2. Rewrite Git history if needed (`git filter-repo` or BFG).
3. Update `.env` locally.

This repository's training script now requires `ROBOFLOW_API_KEY` from the
environment and no longer contains a hard-coded key.

---

## 6. Troubleshooting

- **422 "required gauge points were not detected"**: frame the whole dial,
  improve lighting, reduce motion blur, or retrain with more images.
- **`Invalid image format`**: upload a JPEG or PNG file.
- **Slow first request**: the model is loaded once per worker on first use.
- **Out-of-range readings**: check `clockwise` and `min_value`/`max_value`.
- **Missing model weights**: place `gauge.pt` or the OpenVINO export in the
  model directory.

---

## License

Add your preferred open-source license before publishing.