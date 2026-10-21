#!/usr/bin/env python3
"""Run the full pipeline on a few test images and save annotations plus CSV."""
import argparse
import csv
import logging
from pathlib import Path

import cv2

from src.config import ROOT, load_config
from src.pipeline import RealtimePipeline
from src.utils import FPSCounter, draw_status

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(ROOT / "data/assignment1_clean/test/images"),
        help="image or folder of images",
    )
    parser.add_argument("--output", default=str(ROOT / "results/demo"))
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--detector", choices=("y11n", "y11s"))
    parser.add_argument("--classifier", choices=("sign_n", "sign_s"))
    parser.add_argument("--device", help="auto, cpu, or a CUDA device")
    args = parser.parse_args()
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_config()
    pipeline = RealtimePipeline(
        detector_variant=args.detector,
        classifier_variant=args.classifier,
        device=args.device or config.pipeline.get("device", "auto"),
    )
    source = Path(args.source)
    if source.is_dir():
        images = sorted(path for path in source.iterdir()
                        if path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        images = [source]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "detections.csv"
    fps_counter = FPSCounter()
    rows = []
    stop_frames = 0
    processed = 0

    for image_path in images:
        if processed >= args.max_images:
            break
        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.error("Could not read image: %s", image_path)
            continue
        result = pipeline.process_frame(frame)
        stop_frames += int(result.stop)
        annotated = result.image
        draw_status(annotated, result.stop, fps_counter.tick())
        output_path = output / f"{image_path.stem}.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save annotated image: {output_path}")
        if result.objects:
            for item in result.objects:
                rows.append([
                    image_path.name,
                    item.detection.cls_name,
                    f"{item.detection.conf:.4f}",
                    item.extra_info,
                    "" if item.geometry_distance_m is None else f"{item.geometry_distance_m:.3f}",
                    "" if item.depth_distance_m is None else f"{item.depth_distance_m:.3f}",
                    int(result.stop),
                ])
        else:
            rows.append([image_path.name, "", "", "", "", "", int(result.stop)])
        logging.info("%s: %s", image_path.name,
                     "STOP: " + ", ".join(result.reasons) if result.stop else "GO")
        processed += 1

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "filename", "class", "conf", "extra_info",
            "geo_dist_m", "depth_dist_m", "stop_flag",
        ])
        writer.writerows(rows)
    print(f"Processed {processed} images; STOP frames: {stop_frames}; outputs: {output}")


if __name__ == "__main__":
    main()
