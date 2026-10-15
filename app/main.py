from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import gauge_detection

app = FastAPI(
    title="Gauge Detection API",
    version="2.0.0",
    description="YOLO-based analog gauge reading service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    gauge_detection.router,
    prefix="/api/v1",
)

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")


@app.get("/")
def root():
    return {"message": "Gauge Detection API running", "ui": "/ui/"}