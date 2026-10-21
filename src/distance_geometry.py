"""Pinhole-model distance estimation from configured object dimensions."""
import argparse
import math

from src.config import ROOT, load_config


def calibrate_focal(pixel_height, real_height_m, distance_m):
    """Return focal length in pixels from a known object and measured range."""
    if real_height_m <= 0 or distance_m <= 0:
        raise ValueError("Object height and distance must be positive")
    return pixel_height * distance_m / real_height_m


class GeometryDistance:
    def __init__(self, camera=None, object_sizes=None):
        if camera is None or object_sizes is None:
            config = load_config()
            camera = config.camera if camera is None else camera
            object_sizes = config.object_sizes if object_sizes is None else object_sizes
        self.camera = camera
        self.object_sizes = object_sizes

    def focal_px(self, img_w, axis="x", img_h=None):
        if img_w <= 0 or (axis == "y" and (img_h is None or img_h <= 0)):
            raise ValueError("Image dimensions must be positive")
        camera_value = self.camera.get("fx_px" if axis == "x" else "fy_px")
        calibration_dimension = self.camera.get(
            "image_width_px" if axis == "x" else "image_height_px"
        )
        current_dimension = img_w if axis == "x" else img_h
        if camera_value:
            return float(camera_value) * (
                current_dimension / float(calibration_dimension)
                if calibration_dimension
                else 1.0
            )
        if axis == "y":
            return self.focal_px(img_w, "x")
        sensor_width = self.camera.get("sensor_width_mm")
        focal_mm = self.camera.get("focal_length_mm")
        if sensor_width and focal_mm:
            return float(focal_mm) / float(sensor_width) * img_w
        hfov = self.camera.get("hfov_deg")
        if not hfov or not 0 < float(hfov) < 180:
            raise ValueError("Set fx_px, sensor/focal lengths, or a valid hfov_deg in camera.yaml")
        return (img_w / 2) / math.tan(math.radians(float(hfov)) / 2)

    def intrinsics(self, img_w, img_h):
        camera_width = self.camera.get("image_width_px")
        camera_height = self.camera.get("image_height_px")
        cx = self.camera.get("cx_px")
        cy = self.camera.get("cy_px")
        if cx is None:
            cx = img_w / 2
        elif camera_width:
            cx = float(cx) * img_w / float(camera_width)
        if cy is None:
            cy = img_h / 2
        elif camera_height:
            cy = float(cy) * img_h / float(camera_height)
        return [
            [self.focal_px(img_w, "x"), 0.0, float(cx)],
            [0.0, self.focal_px(img_w, "y", img_h), float(cy)],
            [0.0, 0.0, 1.0],
        ]

    def estimate(self, object_key, box, img_w, img_h, dimension="height"):
        """Estimate range; return None when size metadata or the box is unusable."""
        size = self.object_sizes.get(object_key)
        if not isinstance(size, dict):
            return None
        real_dimension = size.get(f"{dimension}_m")
        x1, y1, x2, y2 = box
        pixel_dimension = (y2 - y1) if dimension == "height" else (x2 - x1)
        if real_dimension is None or real_dimension <= 0 or pixel_dimension < 4:
            return None
        focal = (
            self.focal_px(img_w, "y", img_h)
            if dimension == "height"
            else self.focal_px(img_w, "x")
        )
        return float(focal * float(real_dimension) / pixel_dimension)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    calibration = GeometryDistance()
    print(f"Configured camera focal length: {calibration.focal_px(640):.1f}px at 640px width")
