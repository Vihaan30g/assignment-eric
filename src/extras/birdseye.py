"""Click four plane corners, then project the calibrated floor to bird's-eye."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import ROOT, load_config
from src.detect import Detector

HOMOGRAPHY_PATH = ROOT / "configs/homography.yaml"
ORDER_TEXT = "Click: 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left (clockwise from far edge)"


def load_homography_config():
    with HOMOGRAPHY_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid homography config: {HOMOGRAPHY_PATH}")
    return config


def compute_homography(src_pixels, width_m, length_m):
    """Map ordered TL, TR, BR, BL image points to metric plane coordinates."""
    source = np.asarray(src_pixels, dtype=np.float32)
    if source.shape != (4, 2):
        raise ValueError("Exactly four source pixel points are required")
    if width_m <= 0 or length_m <= 0:
        raise ValueError("Plane dimensions must be positive")
    destination = np.asarray(
        [[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Selected points do not define a valid homography")
    return matrix


def bbox_bottom_center_to_ground(bbox, homography):
    """Return a detected bbox's bottom-center coordinate in ground-plane metres."""
    x1, _, x2, y2 = bbox
    point = np.asarray([[(x1 + x2) / 2, y2]], dtype=np.float32)
    x_m, y_m = cv2.perspectiveTransform(point.reshape(1, 1, 2), homography)[0, 0]
    return float(x_m), float(y_m)


def select_four_points(frame):
    """Select and save four corners; `r` resets and `s` saves."""
    selected = []
    window = "Select ground rectangle corners"

    def draw():
        preview = frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(preview, ORDER_TEXT, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "r: reset    s: save (after 4 clicks)    q: cancel",
                    (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(selected, start=1):
            cv2.circle(preview, point, 9, (0, 0, 255), -1)
            cv2.circle(preview, point, 11, (255, 255, 255), 2)
            cv2.putText(preview, str(index), (point[0] + 12, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(selected) == 4:
            cv2.polylines(preview, [np.asarray(selected, dtype=np.int32)], True, (0, 255, 255), 2)
        return preview

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append((x, y))
            cv2.imshow(window, draw())

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    cv2.imshow(window, draw())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            selected.clear()
            cv2.imshow(window, draw())
        elif key == ord("s") and len(selected) == 4:
            cv2.destroyWindow(window)
            return selected
        elif key == ord("q"):
            cv2.destroyWindow(window)
            raise RuntimeError("Point selection cancelled")


def save_homography(points, frame_shape):
    config = load_homography_config()
    width_m = float(config["plane_width_m"])
    length_m = float(config["plane_length_m"])
    matrix = compute_homography(points, width_m, length_m)
    config["src_points_px"] = [[int(x), int(y)] for x, y in points]
    config["reference_size_px"] = [int(frame_shape[1]), int(frame_shape[0])]
    config["homography_px_to_m"] = matrix.tolist()
    with HOMOGRAPHY_PATH.open("w", encoding="utf-8") as stream:
        stream.write("# USER: replace the points by clicking the four corners in homography_ref.jpg.\n")
        yaml.safe_dump(config, stream, sort_keys=False)
    return matrix, config


def image_to_ground_transform(config, image_shape):
    """Scale calibration points to a frame and compute pixel-to-plane H."""
    points = np.asarray(config.get("src_points_px") or [], dtype=np.float32)
    reference_size = config.get("reference_size_px")
    if points.shape != (4, 2) or not reference_size:
        raise ValueError(
            "No four-point calibration is saved. Run:\n"
            "cd ~/assignment-eric && source /home/marsuser/ml_env/bin/activate && "
            "python -m src.extras.birdseye --select-points"
        )
    reference_width, reference_height = reference_size
    width, height = image_shape[1], image_shape[0]
    scaled = points * np.asarray([width / reference_width, height / reference_height], dtype=np.float32)
    return compute_homography(scaled, config["plane_width_m"], config["plane_length_m"])


def automatic_bev(frame, ground_h, grid_spacing_m=0.10):
    """Warp to an automatically bounded metric canvas and draw a 10cm grid."""
    height, width = frame.shape[:2]
    corners_px = np.asarray([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
                            dtype=np.float32)
    corners_m = cv2.perspectiveTransform(corners_px, ground_h)[0]
    if not np.isfinite(corners_m).all():
        raise ValueError("Image corners project to invalid metric coordinates")
    minimum = corners_m.min(axis=0)
    maximum = corners_m.max(axis=0)
    extent = maximum - minimum
    if np.any(extent <= 0) or np.any(extent > 100):
        raise ValueError(f"Unreasonable projected floor extent ({extent[0]:.2f} x {extent[1]:.2f} m)")
    scale = 50.0
    translate = np.asarray([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    shift_h = translate @ ground_h
    canvas_size = (int(np.ceil(extent[0] * scale)) + 1, int(np.ceil(extent[1] * scale)) + 1)
    bev = cv2.warpPerspective(frame, np.diag([scale, scale, 1.0]) @ shift_h, canvas_size)
    if grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be positive")
    grid_step_px = max(1, int(round(grid_spacing_m * scale)))
    for x in range(0, canvas_size[0], grid_step_px):
        cv2.line(bev, (x, 0), (x, canvas_size[1] - 1), (60, 150, 60), 1)
        if x % int(round(0.5 * scale)) == 0:
            label = f"{minimum[0] + x / scale:.1f}m"
            cv2.putText(bev, label, (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255, 255, 255), 1)
    for y in range(0, canvas_size[1], grid_step_px):
        cv2.line(bev, (0, y), (canvas_size[0] - 1, y), (60, 150, 60), 1)
        if y % int(round(0.5 * scale)) == 0:
            label = f"{minimum[1] + y / scale:.1f}m"
            cv2.putText(bev, label, (2, max(12, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    return bev, shift_h, minimum, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="image, video, or camera index (defaults to reference image for calibration)")
    parser.add_argument("--select-points", action="store_true", help="click and save reference rectangle corners")
    parser.add_argument("--output", default=str(ROOT / "results/extras/birdseye_real.png"))
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    config = load_homography_config()
    calibrated_frame = None
    if args.select_points:
        reference_image = ROOT / config["reference_image"]
        calibrated_frame = cv2.imread(str(reference_image))
        if calibrated_frame is None:
            raise OSError(f"Could not read calibration image: {reference_image}")
        print(ORDER_TEXT)
        points = select_four_points(calibrated_frame)
        matrix, config = save_homography(points, calibrated_frame.shape)
        mapped = cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float32).reshape(1, 4, 2), matrix
        )[0]
        target = np.asarray(
            [[0, 0], [config["plane_width_m"], 0],
             [config["plane_width_m"], config["plane_length_m"]],
             [0, config["plane_length_m"]]],
            dtype=np.float32,
        )
        errors = np.linalg.norm(mapped - target, axis=1)
        print("Reprojection errors (m):", ", ".join(f"{error:.6f}" for error in errors))
        if not np.allclose(mapped, target, atol=1e-4):
            raise RuntimeError("Homography reprojection check failed")
        print(f"Saved homography from {reference_image} into {HOMOGRAPHY_PATH}")
    elif not args.source:
        parser.error("--source is required unless --select-points is used")

    camera = load_config().camera
    source_value = args.source or config["reference_image"]
    source_path = Path(source_value)
    is_image = source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if is_image:
        frame = calibrated_frame if args.select_points and not args.source else cv2.imread(str(source_path))
        if frame is None:
            raise OSError(f"Could not read image: {source_path}")
    else:
        capture = cv2.VideoCapture(int(source_value) if str(source_value).isdigit() else str(source_value))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {source_value}")
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not read a frame from: {source_value}")
    ground_h = image_to_ground_transform(config, frame.shape)
    bev, shift_h, minimum, scale = automatic_bev(
        frame, ground_h, float(config.get("grid_spacing_m", 0.10))
    )
    pipeline = load_config().pipeline
    detector = Detector(
        ROOT / pipeline["models"]["detectors"][pipeline["models"]["default_detector"]],
        pipeline["class_roles"],
        confidence=float(pipeline["detection"]["confidence"]),
        image_size=int(pipeline["detection"]["image_size"]),
    )
    detections = detector.predict(frame)
    outline = np.asarray(config["src_points_px"], dtype=np.float32)
    rw, rh = config["reference_size_px"]
    outline *= np.asarray([frame.shape[1] / rw, frame.shape[0] / rh], dtype=np.float32)
    original = frame.copy()
    cv2.polylines(original, [outline.astype(np.int32)], True, (0, 0, 255), 3)
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        x_m, y_m = bbox_bottom_center_to_ground(detection.xyxy, ground_h)
        x_px, y_px = (x_m - minimum[0]) * scale, (y_m - minimum[1]) * scale
        if 0 <= x_px < bev.shape[1] and 0 <= y_px < bev.shape[0]:
            cv2.circle(bev, (int(x_px), int(y_px)), 7, (0, 0, 255), -1)
        cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
    bev_panel = cv2.resize(bev, (int(bev.shape[1] * original.shape[0] / bev.shape[0]), original.shape[0]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.hstack((original, bev_panel))):
        raise OSError(f"Could not write output: {output}")
    print(f"Projected visible floor extent: {bev.shape[1] / scale:.2f} x {bev.shape[0] / scale:.2f}m")
    print(f"Detections projected: {len(detections)}; saved {output}")


if __name__ == "__main__":
    main()
