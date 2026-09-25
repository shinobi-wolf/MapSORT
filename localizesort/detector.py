"""Detector submodule. See docs/modules.yaml#Detector."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from .schemas import Detection

class Detector:
    """YOLO-based object detector (Ultralytics YOLOv8).

    Produces per-frame bounding-box detections D_t from the raw RGB frame I_t.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        # Ultralytics treats numpy input as BGR; the pipeline frames are RGB.
        bgr = np.ascontiguousarray(frame[..., ::-1])
        [result] = self.model.predict(
            bgr, conf=self.conf, device=self.device, verbose=False
        )

        return [
            Detection(
                bbox=box.xyxy[0].cpu().numpy(),
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
            )
            for box in result.boxes
        ]
