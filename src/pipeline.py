"""Single-frame processing for detection, trigger classification, and range."""
from dataclasses import dataclass
import logging
from pathlib import Path

from src.config import ROOT, load_config
from src.detect import Detection, Detector
from src.distance_geometry import GeometryDistance
from src.distance_depth import DepthEstimator
from src.sign_classifier import SignClassifier
from src.traffic_light import classify_light_color
from src.utils import display_name, draw_box, distance_color

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectResult:
    detection: Detection
    label: str
    extra_info: str
    geometry_distance_m: float | None
    depth_distance_m: float | None
    is_trigger: bool


@dataclass(frozen=True)
class FrameResult:
    image: object
    stop: bool
    reasons: tuple[str, ...]
    objects: tuple[ObjectResult, ...]


class RealtimePipeline:
    def __init__(
        self,
        detector_variant=None,
        classifier_variant=None,
        device="auto",
        confidence=None,
        pipeline_path=ROOT / "configs/pipeline.yaml",
        camera_path=ROOT / "configs/camera.yaml",
        object_sizes_path=ROOT / "configs/object_sizes.yaml",
    ):
        self.config = load_config(pipeline_path, camera_path, object_sizes_path)
        model_config = self.config.pipeline["models"]
        detector_variant = detector_variant or model_config["default_detector"]
        classifier_variant = classifier_variant or model_config["default_classifier"]
        detector_weights = self._weight_path(model_config["detectors"], detector_variant)
        classifier_weights = self._weight_path(model_config["classifiers"], classifier_variant)
        settings = self.config.pipeline["detection"]
        self.detector = Detector(
            detector_weights,
            self.config.roles,
            confidence if confidence is not None else float(settings["confidence"]),
            int(settings["image_size"]),
            device,
        )
        self.sign_classifier = SignClassifier(
            classifier_weights,
            self.config.pipeline["sign_labels"],
            device,
        )
        self.geometry = GeometryDistance(self.config.camera, self.config.object_sizes)
        self.depth_estimator = None
        if self.config.pipeline.get("depth_enabled", True):
            depth_config = self.config.pipeline["models"]
            self.depth_estimator = DepthEstimator(
                variant=depth_config["depth_model"],
                device=device,
                scale_factor=self.config.camera.get("depth_scale", 1.0),
            )

    @staticmethod
    def _weight_path(variants, variant):
        if variant not in variants:
            raise ValueError(f"Unknown model variant {variant!r}; choose from {', '.join(variants)}")
        path = Path(variants[variant])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
        return path

    @staticmethod
    def _crop(frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(x2)), min(height, int(y2))
        return frame[top:bottom, left:right]

    @staticmethod
    def _object_size_key(detection, extra_info):
        if detection.role == "sign_board":
            if extra_info == "stop":
                return "stop_sign"
            if extra_info == "barrier ahead":
                return "barrier_ahead_sign"
            return "sign_board"
        return {
            "barrier": "barrier",
            "cone": "cone",
            "traffic_light": "traffic_light",
        }.get(detection.role)

    @staticmethod
    def _object_label(detection, extra_info):
        if detection.role == "traffic_light":
            if extra_info and extra_info != "unknown":
                return f"{extra_info.title()} Light"
        elif detection.role == "sign_board":
            if extra_info == "stop":
                return "Stop Sign"
            if extra_info == "barrier ahead":
                return "Barrier Ahead Sign"
        return display_name(detection.cls_name)

    @staticmethod
    def _format_label(title, geometry_distance, depth_distance):
        if geometry_distance is not None and depth_distance is not None:
            return f"{title}, Geo {geometry_distance:.1f}m | Depth {depth_distance:.1f}m"
        distance = depth_distance if depth_distance is not None else geometry_distance
        return f"{title}, {distance:.1f}m" if distance is not None else title

    def process_frame(self, frame):
        detections = self.detector.predict(frame)
        staged = []
        reasons = []
        for detection in detections:
            crop = self._crop(frame, detection.xyxy)
            extra_info = ""
            if detection.role == "traffic_light":
                extra_info, _ = classify_light_color(crop)
            elif detection.role == "sign_board":
                extra_info, _ = self.sign_classifier.classify(crop)

            trigger = False
            reason = None
            if detection.role == "barrier":
                trigger, reason = True, "barrier"
            elif detection.role == "traffic_light" and extra_info == "red":
                trigger, reason = True, "red light"
            elif detection.role == "sign_board" and extra_info == "stop":
                trigger, reason = True, "stop sign"
            elif detection.role == "sign_board" and extra_info == "barrier ahead":
                trigger, reason = True, "barrier ahead sign"
            if reason and reason not in reasons:
                reasons.append(reason)
            staged.append((detection, extra_info, trigger))

        depth_map = None
        if any(trigger for _, _, trigger in staged) and self.depth_estimator is not None:
            try:
                depth_map = self.depth_estimator.predict(
                    frame, self.geometry.intrinsics(frame.shape[1], frame.shape[0])
                )
            except Exception:
                LOGGER.exception("Depth inference failed for a trigger frame; geometry remains available")

        objects = []
        annotated = frame.copy()
        image_height, image_width = frame.shape[:2]
        for detection, extra_info, trigger in staged:
            size_key = self._object_size_key(detection, extra_info)
            geometry_distance = (
                self.geometry.estimate(size_key, detection.xyxy, image_width, image_height)
                if size_key
                else None
            )
            depth_distance = None
            if trigger and depth_map is not None:
                depth_distance = self.depth_estimator.distance_for_bbox(depth_map, detection.xyxy)
            title = self._object_label(detection, extra_info)
            label = self._format_label(title, geometry_distance, depth_distance)
            color_distance = depth_distance if depth_distance is not None else geometry_distance
            draw_box(annotated, detection.xyxy, label, distance_color(color_distance))
            objects.append(
                ObjectResult(
                    detection=detection,
                    label=label,
                    extra_info=extra_info,
                    geometry_distance_m=geometry_distance,
                    depth_distance_m=depth_distance,
                    is_trigger=trigger,
                )
            )
        return FrameResult(
            image=annotated,
            stop=bool(reasons),
            reasons=tuple(reasons),
            objects=tuple(objects),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("y11n", "y11s"))
    parser.add_argument("--classifier", choices=("sign_n", "sign_s"))
    parser.parse_args()
