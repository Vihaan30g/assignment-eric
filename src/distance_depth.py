"""Metric monocular depth estimation using the configured UniDepth model."""
import cv2
import numpy as np
import torch

from src.detect import resolve_device


class DepthEstimator:
    def __init__(self, variant="vits14", device="auto", scale_factor=1.0):
        from unidepth.models import UniDepthV2

        self.device = resolve_device(device)
        self.scale_factor = float(scale_factor)
        if self.scale_factor <= 0:
            raise ValueError("Depth scale factor must be positive")
        model_name = f"lpiccinelli/unidepth-v2-{variant}"
        self.model = UniDepthV2.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, frame_bgr, intrinsics=None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        if intrinsics is not None:
            matrix = torch.tensor(intrinsics, dtype=torch.float32, device=self.device)
            output = self.model.infer(image, matrix)
        else:
            output = self.model.infer(image)
        depth_tensor = output["depth"].float()
        while depth_tensor.ndim > 2 and depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor[0]
        depth = depth_tensor.cpu().numpy()
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, received shape {depth.shape}")
        if depth.shape != frame_bgr.shape[:2]:
            depth = cv2.resize(
                depth,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth * self.scale_factor

    @staticmethod
    def distance_for_bbox(depth_map, bbox, shrink=0.5):
        if depth_map is None or depth_map.ndim != 2:
            raise ValueError("Depth map must be a 2D array")
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = (x2 - x1) * shrink, (y2 - y1) * shrink
        map_height, map_width = depth_map.shape
        left = max(0, int(center_x - width / 2))
        right = min(map_width, int(center_x + width / 2) + 1)
        top = max(0, int(center_y - height / 2))
        bottom = min(map_height, int(center_y + height / 2) + 1)
        roi = depth_map[top:bottom, left:right]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.median(valid)) if valid.size else None


class UniDepthEstimator(DepthEstimator):
    """Backwards-compatible name retained for existing scripts."""


def box_depth(depth_map, box, shrink=0.5):
    return DepthEstimator.distance_for_bbox(depth_map, box, shrink)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="vits14")
    parser.parse_args()
