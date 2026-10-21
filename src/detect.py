"""YOLO detector wrapper returning stable, typed detections."""
from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str
    role: str


def resolve_device(device):
    if device and device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Detector:
    def __init__(self, weights, class_roles, confidence=0.35, image_size=640, device="auto"):
        self.model = YOLO(str(weights))
        self.class_roles = class_roles
        self.confidence = confidence
        self.image_size = image_size
        self.device = resolve_device(device)
        self.names = self.model.names

    @torch.inference_mode()
    def predict(self, frame):
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        names = result.names
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = names[class_id]
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    xyxy=coordinates,
                    conf=float(box.conf.item()),
                    cls_id=class_id,
                    cls_name=class_name,
                    role=self.class_roles.get(class_name, "other"),
                )
            )
        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="trained detector checkpoint")
    parser.parse_args()
