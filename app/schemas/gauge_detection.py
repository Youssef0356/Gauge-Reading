from typing import List, Optional

from pydantic import BaseModel


class GaugeDetectionResponse(BaseModel):
    status: str
    value: Optional[float] = None
    detected_value: Optional[float] = None
    unit: str = ""
    fraction: Optional[float] = None
    confidence: float = 0.0
    angles: Optional[dict] = None
    detections: List[dict] = []
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_in_range: Optional[bool] = None
    condition: Optional[str] = None
    equipment_id: Optional[str] = None