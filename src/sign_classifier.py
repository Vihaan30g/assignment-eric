"""YOLO classification wrapper for sign-board crops."""
import torch
from ultralytics import YOLO

from src.detect import resolve_device


class SignClassifier:
    def __init__(self, weights, sign_labels, device="auto"):
        self.model = YOLO(str(weights))
        self.sign_labels = sign_labels
        self.device = resolve_device(device)
        self.names = self.model.names

    @torch.inference_mode()
    def classify(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Cannot classify an empty sign crop")
        result = self.model.predict(crop_bgr, device=self.device, verbose=False)[0]
        probabilities = result.probs
        class_id = int(probabilities.top1)
        raw_label = result.names[class_id]
        label = self.sign_labels.get(raw_label, raw_label.replace("_", " "))
        return label, float(probabilities.top1conf.item())

    def predict(self, crop_bgr):
        """Compatibility alias for callers that used the earlier wrapper."""
        return self.classify(crop_bgr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="trained sign classifier checkpoint")
    parser.parse_args()
