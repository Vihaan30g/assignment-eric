"""Track cone/barrier boxes and compensate global camera motion."""
import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from src.config import ROOT, load_config
from src.detect import Detector

FLOW_CONFIG_PATH = ROOT / "configs/optical_flow.yaml"
TRACKED_ROLES = {"cone", "barrier"}


def load_flow_config():
    with FLOW_CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid optical-flow config: {FLOW_CONFIG_PATH}")
    return config


def features_in_box(gray, bbox):
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 255
    return cv2.goodFeaturesToTrack(
        gray, maxCorners=60, qualityLevel=0.01, minDistance=5, blockSize=5, mask=mask,
    )


def background_features(gray, boxes):
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    height, width = gray.shape[:2]
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            mask,
            (max(0, int(x1)), max(0, int(y1))),
            (min(width - 1, int(x2)), min(height - 1, int(y2))),
            0,
            -1,
        )
    return cv2.goodFeaturesToTrack(
        gray, maxCorners=500, qualityLevel=0.01, minDistance=8, blockSize=7, mask=mask,
    )


def flow_points(previous_gray, gray, points):
    if points is None or len(points) == 0:
        return None, None, None
    current, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3,
    )
    if current is None or status is None:
        return None, None, None
    valid = status.ravel() == 1
    previous_points = points[valid].reshape(-1, 2)
    current_points = current[valid].reshape(-1, 2)
    return previous_points, current_points, current_points - previous_points


def intersection_over_union(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return overlap / max(1e-6, area_first + area_second - overlap)


def select_manual_roi(start_frame):
    roi = cv2.selectROI(
        f"Select moving object at frame {start_frame}",
        start_frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(f"Select moving object at frame {start_frame}")
    x, y, width, height = map(int, roi)
    if width <= 0 or height <= 0:
        raise RuntimeError("No ROI selected")
    return (x, y, x + width, y + height)


def parse_roi(text):
    try:
        x, y, width, height = map(int, text.split(","))
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from error
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI coordinates must be non-negative with positive dimensions")
    return x, y, x + width, y + height


def detector_for_config(device):
    config = load_config().pipeline
    variant = config["models"]["default_detector"]
    settings = load_flow_config()
    return Detector(
        ROOT / config["models"]["detectors"][variant],
        config["class_roles"],
        confidence=float(settings.get("detector_confidence", 0.35)),
        image_size=int(settings.get("image_size", 640)),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="video path or camera index")
    parser.add_argument("--output", default=str(ROOT / "results/extras/optical_flow_real.mp4"))
    parser.add_argument("--frame-output", default=str(ROOT / "results/extras/optical_flow_real.png"))
    parser.add_argument("--csv", default=str(ROOT / "results/extras/optical_flow_real.csv"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--every-n", type=int, default=3, help="run detector every N frames")
    parser.add_argument("--select-roi", action="store_true",
                        help="manually select an ROI at --start-frame instead of running detection")
    parser.add_argument("--roi", type=parse_roi, help="manual ROI as x,y,width,height")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="first frame to process (use with --roi or --select-roi)")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.every_n < 1:
        parser.error("--every-n must be at least 1")
    if args.start_frame < 0:
        parser.error("--start-frame cannot be negative")
    if args.select_roi and args.roi is not None:
        parser.error("use either --select-roi or --roi, not both")
    camera = load_config().camera
    focal_px = camera.get("fx_px")
    settings = load_flow_config()
    threshold = float(settings["moving_threshold_px_per_sec"])
    if threshold < 0:
        raise ValueError("moving_threshold_px_per_sec cannot be negative")

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Video source did not provide a valid FPS")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    first_ok, first_frame = cap.read()
    if not first_ok:
        raise RuntimeError(f"Could not read video frame {args.start_frame}")
    detector = detector_for_config(args.device)

    manual_bbox = args.roi
    if args.select_roi:
        manual_bbox = select_manual_roi(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    output = Path(args.output)
    frame_output = Path(args.frame_output)
    csv_path = Path(args.csv)
    for path in (output, frame_output, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    previous_gray = None
    tracks = []
    next_object_id = 1
    previous_background = None
    frame_index = args.start_frame
    preview_saved = False
    fallback_preview = None
    manual_mode = manual_bbox is not None
    mode = "manual" if manual_mode else "auto"
    print(f"Optical-flow mode: {mode}; starting at frame {frame_index}; detector interval={args.every_n}")
    saw_tracked_object = False
    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
        csv_writer = csv.writer(csv_stream)
        csv_writer.writerow(["frame", "object_id", "vx", "vy", "speed", "moving"])
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection_refresh = frame_index % args.every_n == 0
                detections = (
                    [d for d in detector.predict(frame) if d.role in TRACKED_ROLES]
                    if detection_refresh and not manual_mode
                    else []
                )
                background_delta = np.zeros(2, dtype=np.float32)
                if previous_gray is not None and previous_background is not None:
                    _, background_current, background_motion = flow_points(
                        previous_gray, gray, previous_background
                    )
                    if background_motion is not None and len(background_motion) >= 3:
                        background_delta = np.median(background_motion, axis=0)
                        previous_background = background_current.reshape(-1, 1, 2)
                    else:
                        previous_background = None

                updated_tracks = []
                for track in tracks:
                    velocity = np.zeros(2, dtype=np.float32)
                    next_points = None
                    raw_delta = np.zeros(2, dtype=np.float32)
                    if previous_gray is not None:
                        old_points, current_points, raw_motion = flow_points(
                            previous_gray, gray, track["points"]
                        )
                        if raw_motion is not None and len(raw_motion):
                            raw_delta = np.median(raw_motion, axis=0)
                            corrected = raw_motion - background_delta
                            velocity = np.median(corrected, axis=0) * fps
                            next_points = current_points.reshape(-1, 1, 2)
                    x1, y1, x2, y2 = track["bbox"]
                    predicted_box = (
                        x1 + float(raw_delta[0]), y1 + float(raw_delta[1]),
                        x2 + float(raw_delta[0]), y2 + float(raw_delta[1]),
                    )
                    updated_tracks.append({
                        **track,
                        "bbox": predicted_box,
                        "points": next_points,
                        "velocity": velocity,
                        "matched": False,
                    })

                if detection_refresh and manual_bbox is None:
                    for detection in detections:
                        box = detection.xyxy
                        matches = [
                            (intersection_over_union(track["bbox"], box), index)
                            for index, track in enumerate(updated_tracks)
                            if not track["matched"] and track["class_name"] == detection.cls_name
                        ]
                        best_iou, best_index = max(matches, default=(0.0, -1))
                        if best_iou >= 0.1:
                            updated_tracks[best_index]["bbox"] = box
                            updated_tracks[best_index]["matched"] = True
                            updated_tracks[best_index]["last_detection_frame"] = frame_index
                        else:
                            updated_tracks.append({
                                "object_id": next_object_id,
                                "class_name": detection.cls_name,
                                "bbox": box,
                                "points": None,
                                "velocity": np.zeros(2, dtype=np.float32),
                                "matched": True,
                                "last_detection_frame": frame_index,
                            })
                            next_object_id += 1

                    updated_tracks = [
                        track for track in updated_tracks
                        if frame_index - track["last_detection_frame"] <= args.every_n * 5
                    ]

                if manual_mode and not tracks:
                    updated_tracks = [{
                        "object_id": next_object_id,
                        "class_name": "manual ROI",
                        "bbox": manual_bbox,
                        "points": None,
                        "velocity": np.zeros(2, dtype=np.float32),
                        "matched": True,
                        "last_detection_frame": frame_index,
                    }]
                    next_object_id += 1

                saw_tracked_object = saw_tracked_object or bool(updated_tracks)
                for track in updated_tracks:
                    track["points"] = features_in_box(gray, track["bbox"])
                    vx, vy = map(float, track["velocity"])
                    speed = float(np.hypot(vx, vy))
                    moving = speed > threshold
                    x1, y1, x2, y2 = map(int, track["bbox"])
                    color = (0, 0, 255) if moving else (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {track['object_id']} {speed:.1f}px/s {'MOVING' if moving else 'STATIC'}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    label_y = max(22, y1 - 6)
                    cv2.rectangle(frame, (x1, label_y - label_height - 5),
                                  (x1 + label_width + 4, label_y + 3), (20, 20, 20), -1)
                    cv2.putText(frame, label, (x1 + 2, label_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    arrow_end = (int(center[0] + vx * 0.15), int(center[1] + vy * 0.15))
                    if speed > 0:
                        cv2.arrowedLine(frame, center, arrow_end, (255, 100, 0), 2)
                    csv_writer.writerow([
                        frame_index, track["object_id"], f"{vx:.3f}", f"{vy:.3f}",
                        f"{speed:.3f}", int(moving),
                    ])
                    print(
                        f"frame {frame_index}, object {track['object_id']}: "
                        f"vx={vx:.1f}, vy={vy:.1f}, speed={speed:.1f} px/s "
                        f"{'MOVING' if moving else 'STATIC'}"
                    )
                if focal_px is not None:
                    cv2.putText(frame, f"Camera fx: {float(focal_px):.1f}px",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if updated_tracks and fallback_preview is None:
                    fallback_preview = frame.copy()
                if not preview_saved and any(
                    np.hypot(*track["velocity"]) > threshold for track in updated_tracks
                ):
                    if not cv2.imwrite(str(frame_output), frame):
                        raise OSError(f"Could not save representative frame: {frame_output}")
                    preview_saved = True
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise OSError(f"Could not create video output: {output}")
                writer.write(frame)
                if not args.no_display:
                    cv2.imshow("Global-motion-compensated optical flow", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                boxes = [track["bbox"] for track in updated_tracks]
                previous_background = background_features(gray, boxes)
                previous_gray = gray
                tracks = updated_tracks
                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()
    if not preview_saved and fallback_preview is not None:
        if not cv2.imwrite(str(frame_output), fallback_preview):
            raise OSError(f"Could not save representative frame: {frame_output}")
    print(f"Processed frames through {frame_index - 1}; mode={mode}; output={output}; CSV={csv_path}")
    if not saw_tracked_object:
        print("No cone/barrier detections were found anywhere in the video.")


if __name__ == "__main__":
    main()
