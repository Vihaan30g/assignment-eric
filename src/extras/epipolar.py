"""Stereo disparity/depth utilities and a StereoSGBM pair demo."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import ROOT, load_config


def disparity_to_depth(disparity_px, f_px, baseline_m):
    """Convert rectified stereo disparity in pixels to depth in metres."""
    disparity = np.asarray(disparity_px, dtype=np.float32)
    if f_px <= 0 or baseline_m <= 0:
        raise ValueError("Focal length and baseline must be positive")
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = float(f_px) * float(baseline_m) / disparity[valid]
    return float(depth) if depth.ndim == 0 else depth


def depth_to_disparity(depth_m, f_px, baseline_m):
    """Convert depth in metres to rectified stereo disparity in pixels."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if f_px <= 0 or baseline_m <= 0 or np.any(depth <= 0):
        raise ValueError("Depth, focal length, and baseline must be positive")
    disparity = float(f_px) * float(baseline_m) / depth
    return float(disparity) if disparity.ndim == 0 else disparity


def camera_focal_px(camera, image_width):
    """Read calibrated focal length or derive it from camera.yaml's FOV."""
    fx = camera.get("fx_px")
    calibration_width = camera.get("image_width_px")
    if fx:
        return float(fx) * (image_width / float(calibration_width) if calibration_width else 1.0)
    sensor_width = camera.get("sensor_width_mm")
    focal_mm = camera.get("focal_length_mm")
    if sensor_width and focal_mm:
        return float(focal_mm) / float(sensor_width) * image_width
    hfov = float(camera.get("hfov_deg", 70.0))
    return image_width / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="rectified left image")
    parser.add_argument("--right", required=True, help="rectified right image")
    parser.add_argument("--baseline-m", type=float, help="stereo baseline; defaults to camera.yaml")
    parser.add_argument("--output", default=str(ROOT / "results/extras/stereo_depth_demo.png"))
    args = parser.parse_args()
    camera = load_config().camera
    baseline = args.baseline_m if args.baseline_m is not None else camera.get("stereo_baseline_m")
    if baseline is None:
        parser.error("Set stereo_baseline_m in configs/camera.yaml or pass --baseline-m")
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)
    if left is None or right is None:
        raise OSError("Could not read both stereo images")
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("Stereo images must have the same dimensions")
    height, width = left.shape[:2]
    focal_px = camera_focal_px(camera, width)
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    depth = disparity_to_depth(disparity, focal_px, baseline)
    valid = np.isfinite(depth)
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    if np.any(valid):
        normalized = np.zeros_like(depth, dtype=np.uint8)
        lower, upper = np.percentile(depth[valid], (2, 98))
        if upper > lower:
            normalized[valid] = (
                (np.clip(depth[valid], lower, upper) - lower) * (255.0 / (upper - lower))
            ).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), colored):
        raise OSError(f"Could not save depth map: {output}")
    print(f"Focal length: {focal_px:.1f}px; baseline: {baseline:g}m; saved {output}")


if __name__ == "__main__":
    main()
