import asyncio

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from app.schemas.gauge_detection import GaugeDetectionResponse
from app.services.gauge_detection.detector import GaugeDetector, GaugeReadError, get_detector

router = APIRouter(
    prefix="/gauge",
    tags=["gauge detection"],
)


@router.post("/detect", response_model=GaugeDetectionResponse)
async def detect_gauge(
    image: UploadFile = File(...),
    min_value: float = Query(0.0, description="Gauge minimum value"),
    max_value: float = Query(100.0, description="Gauge maximum value"),
    unit: str = Query("bar", description="Gauge unit"),
    clockwise: bool = Query(True, description="Whether the needle moves clockwise"),
):
    detector = get_detector()

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    try:
        frame = detector.decode_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}")

    try:
        result = await asyncio.to_thread(
            detector.read,
            frame,
            min_value=min_value,
            max_value=max_value,
            unit=unit,
            clockwise=clockwise,
        )
    except GaugeReadError as exc:
        missing = ", ".join(sorted(exc.missing))
        raise HTTPException(
            status_code=422,
            detail=(
                "Analysis failed: required gauge points were not detected "
                f"({missing}). Frame the whole dial with the needle visible and retry."
            ),
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "Invalid gauge angular span":
            detail = "Could not compute the gauge angle."
        elif detail == "Needle tip is outside the configured MIN/MAX gauge arc":
            detail = "The detected needle is outside the configured MIN/MAX arc."
        elif detail == "max_value must be greater than min_value":
            detail = "Invalid range: max_value must be greater than min_value."
        raise HTTPException(status_code=422, detail=detail)

    value = float(result["value"])
    in_range = min_value <= value <= max_value
    result.update(
        {
            "detected_value": value,
            "detections": [],
            "min_value": min_value,
            "max_value": max_value,
            "is_in_range": in_range,
            "condition": "normal" if in_range else "out_of_range",
        }
    )
    return result