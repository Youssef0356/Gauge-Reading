import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class GaugeReadError(Exception):
    """Raised when required gauge geometry (center/min/max/tip) wasn't found in a frame."""

    def __init__(self, message: str, missing: set | None = None):
        super().__init__(message)
        self.missing = missing or set()


class GaugeDetector:
    """
    Stateless (per-gauge) analog gauge reader. The YOLO/OpenVINO model is loaded
    ONCE and shared through get_detector(). min/max/unit/clockwise are supplied
    per call, since one instance serves every gauge.
    """

    REQUIRED_CLASSES = {"center", "min", "max", "tip"}
    GAUGE_CLASS = "gauge"
    FALLBACK_CONFIDENCE_THRESHOLD = 0.12

    def __init__(
        self, model_path: str | Path | None = None, confidence_threshold: float = 0.25
    ):
        if model_path is None:
            base = Path(__file__).resolve().parent / "models"
            preferred = base / "gauge_openvino_model"
            model_path = preferred if preferred.exists() else base / "gauge.pt"
        self.model = YOLO(str(model_path), task="detect")
        self.confidence_threshold = confidence_threshold
        logger.info("GaugeDetector loaded model from %s", model_path)

    def analyze_frame(self, frame: np.ndarray) -> dict[str, Any]:
        # Keep lower-confidence candidates available for the crop/fill retry;
        # _extract_points still applies the normal 0.25 threshold first.
        results = self.model(
            frame, verbose=False, conf=self.FALLBACK_CONFIDENCE_THRESHOLD
        )
        points = self._extract_points(results)

        # A phone photo often contains a lot of background, making the four
        # point classes too small for a single full-frame inference. If the
        # model can locate the gauge itself, run a second inference on a
        # padded crop and use it to fill only the missing point classes.
        if self.REQUIRED_CLASSES - set(points):
            crop = self._gauge_crop(frame, results)
            if crop is not None:
                crop_frame, offset = crop
                crop_results = self.model(
                    crop_frame,
                    verbose=False,
                    conf=self.FALLBACK_CONFIDENCE_THRESHOLD,
                )
                crop_points = self._extract_points(
                    crop_results,
                    confidence_threshold=self.FALLBACK_CONFIDENCE_THRESHOLD,
                    offset=offset,
                )
                for class_name, candidate in crop_points.items():
                    current = points.get(class_name)
                    if (
                        current is None
                        or candidate["confidence"] > current["confidence"]
                    ):
                        points[class_name] = candidate

        # Finally, recover low-confidence points from the original frame. This
        # is deliberately a fill-only pass so it cannot replace a confident
        # point with a weaker false positive.
        missing = self.REQUIRED_CLASSES - set(points)
        if missing:
            low_confidence_points = self._extract_points(
                results, confidence_threshold=self.FALLBACK_CONFIDENCE_THRESHOLD
            )
            for class_name in missing:
                candidate = low_confidence_points.get(class_name)
                if candidate is not None:
                    points[class_name] = candidate

        missing = self.REQUIRED_CLASSES - set(points.keys())
        if missing:
            logger.warning(
                "Gauge detection incomplete: missing=%s detected=%s frame_shape=%s",
                sorted(missing),
                self._summarize_detections(results),
                frame.shape,
            )
            raise GaugeReadError(
                f"Could not detect required gauge points: {', '.join(sorted(missing))}",
                missing=missing,
            )

        center = points["center"]["point"]
        angles = {
            "min": self._angle_between(center, points["min"]["point"]),
            "max": self._angle_between(center, points["max"]["point"]),
            "tip": self._angle_between(center, points["tip"]["point"]),
        }
        confidence = float(np.mean([p["confidence"] for p in points.values()]))
        return {
            "points": points,
            "angles": angles,
            "center": center,
            "confidence": confidence,
        }

    def _summarize_detections(self, results) -> dict[str, list[float]]:
        """Summarize raw candidates so failed photos are diagnosable in logs."""
        summary: dict[str, list[float]] = {}
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                if class_name not in {*self.REQUIRED_CLASSES, self.GAUGE_CLASS}:
                    continue
                summary.setdefault(class_name, []).append(round(float(box.conf[0]), 3))
        return summary

    def compute_value(
        self,
        angles: dict[str, float],
        min_value: float,
        max_value: float,
        clockwise: bool = True,
    ) -> dict[str, float]:
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value")

        fraction = self._calculate_fraction(
            min_angle=angles["min"],
            max_angle=angles["max"],
            tip_angle=angles["tip"],
            clockwise=clockwise,
        )
        value = min_value + fraction * (max_value - min_value)
        return {"fraction": fraction, "value": value}

    def read(
        self,
        frame: np.ndarray,
        min_value: float,
        max_value: float,
        unit: str = "",
        clockwise: bool = True,
    ) -> dict[str, Any]:
        analysis = self.analyze_frame(frame)
        result = self.compute_value(analysis["angles"], min_value, max_value, clockwise)
        return {
            "status": "success",
            "value": round(result["value"], 3),
            "unit": unit,
            "fraction": round(result["fraction"], 6),
            "confidence": round(analysis["confidence"], 3),
            "angles": {k: round(v, 2) for k, v in analysis["angles"].items()},
        }

    @staticmethod
    def decode_image(contents: bytes) -> np.ndarray:
        np_image = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image format")
        return frame

    def _extract_points(
        self,
        results,
        confidence_threshold: float | None = None,
        offset: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        point_offset = offset if offset is not None else np.zeros(2, dtype=np.float64)
        detections: dict[str, list[dict[str, Any]]] = {
            "center": [],
            "min": [],
            "max": [],
            "tip": [],
        }
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < threshold:
                    continue
                class_name = self.model.names[int(box.cls[0])]
                if class_name not in self.REQUIRED_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                point = (
                    np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
                    + point_offset
                )
                detections[class_name].append(
                    {
                        "point": point,
                        "confidence": confidence,
                        "box": [
                            x1 + point_offset[0],
                            y1 + point_offset[1],
                            x2 + point_offset[0],
                            y2 + point_offset[1],
                        ],
                    }
                )
        return {
            name: max(candidates, key=lambda c: c["confidence"])
            for name, candidates in detections.items()
            if candidates
        }

    def _gauge_crop(self, frame: np.ndarray, results):
        """Return a padded gauge crop and its offset in the original frame."""
        candidates = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                if class_name != self.GAUGE_CLASS or confidence < 0.10:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidates.append((confidence, x1, y1, x2, y2))

        if not candidates:
            return None

        _, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])
        height, width = frame.shape[:2]
        padding = max(x2 - x1, y2 - y1) * 0.15
        left = max(0, int(x1 - padding))
        top = max(0, int(y1 - padding))
        right = min(width, int(x2 + padding))
        bottom = min(height, int(y2 + padding))
        if right <= left or bottom <= top:
            return None

        return frame[top:bottom, left:right], np.array([left, top], dtype=np.float64)

    @staticmethod
    def _angle_between(origin: np.ndarray, point: np.ndarray) -> float:
        dx = point[0] - origin[0]
        dy = point[1] - origin[1]
        return float(np.degrees(np.arctan2(dy, dx)) % 360.0)

    @staticmethod
    def _calculate_fraction(
        min_angle: float, max_angle: float, tip_angle: float, clockwise: bool
    ) -> float:
        if clockwise:
            span = (max_angle - min_angle) % 360.0
            tip_offset = (tip_angle - min_angle) % 360.0
        else:
            span = (min_angle - max_angle) % 360.0
            tip_offset = (min_angle - tip_angle) % 360.0

        if span <= 0.001:
            raise ValueError("Invalid gauge angular span")
        if tip_offset > span:
            raise ValueError("Needle tip is outside the configured MIN/MAX gauge arc")

        return float(np.clip(tip_offset / span, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Shared singleton (loaded on first use)
# ---------------------------------------------------------------------------

_detector: "GaugeDetector | None" = None
_detector_lock = threading.Lock()


def get_detector() -> GaugeDetector:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = GaugeDetector()
    return _detector