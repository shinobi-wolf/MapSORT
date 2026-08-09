"""Detector submodule. See docs/modules.yaml#Detector."""

from __future__ import annotations

from ultralytics import YOLO

from .schemas import Detection, Detections, RGBFrame


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

    def __call__(self, frame: RGBFrame) -> Detections:
        [result] = self.model.predict(
            frame.image, conf=self.conf, device=self.device, verbose=False
        )

        detections = [
            Detection(
                detection_id=i,
                bbox=box.xyxy[0].cpu().numpy(),
                frame=frame,
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
            )
            for i, box in enumerate(result.boxes)
        ]

        return Detections(frame_index=frame.frame_index, detections=detections)
