"""Run the configured detector, trigger logic, distance, and ROS pipeline."""
import argparse
import logging
from pathlib import Path
import threading

import cv2

from src.config import ROOT, load_config
from src.pipeline import RealtimePipeline
from src.ros_publisher import StopPublisher
from src.utils import FPSCounter, draw_status

LOGGER = logging.getLogger("src.run_realtime")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


class LatestFrameCapture:
    """Camera reader that keeps only its newest frame to avoid stale latency."""

    def __init__(self, camera_index):
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._sequence = 0
        self._frame = None
        self._ended = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if not ok:
                with self._condition:
                    self._ended = True
                    self._condition.notify_all()
                return
            with self._condition:
                self._sequence += 1
                self._frame = frame
                self._condition.notify_all()

    def read_latest(self, previous_sequence, timeout=2.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != previous_sequence or self._ended,
                timeout=timeout,
            )
            if self._sequence == previous_sequence:
                return None
            return self._sequence, self._frame.copy()

    def release(self):
        self._stop.set()
        self.capture.release()
        self._thread.join(timeout=2.0)


def parse_source(source):
    return int(source) if str(source).isdigit() else source


def process_camera(camera_index, process, max_frames):
    camera = LatestFrameCapture(camera_index)
    processed = 0
    sequence = 0
    try:
        while max_frames is None or processed < max_frames:
            item = camera.read_latest(sequence)
            if item is None:
                if camera._ended:
                    break
                continue
            sequence, frame = item
            if not process(frame):
                break
            processed += 1
    finally:
        camera.release()


def process_stream(source, process, max_frames):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {source}")
    processed = 0
    try:
        while max_frames is None or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if not process(frame):
                break
            processed += 1
    finally:
        capture.release()


def process_images(source, process, max_frames):
    path = Path(source)
    paths = (
        sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
        if path.is_dir()
        else [path]
    )
    processed = 0
    for image_path in paths:
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            LOGGER.error("Could not read image: %s", image_path)
            continue
        if not process(frame):
            break
        processed += 1
        if max_frames is not None and processed >= max_frames:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="camera index, video path, or image/folder path")
    parser.add_argument("--detector", choices=("y11n", "y11s"))
    parser.add_argument("--classifier", choices=("sign_n", "sign_s"))
    parser.add_argument("--device", help="auto, cpu, or a CUDA device (default: configured)")
    parser.add_argument("--conf", type=float, help="detector confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="run headlessly")
    parser.add_argument("--no-ros", action="store_true", help="disable ROS2 publishing")
    parser.add_argument("--save-video", metavar="PATH", help="write annotated frames to an MP4")
    parser.add_argument("--max-frames", type=int, help="stop after this many processed frames")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    source = parse_source(args.source if args.source is not None else config.pipeline["camera_source"])
    device = args.device or config.pipeline.get("device", "auto")
    pipeline = RealtimePipeline(
        detector_variant=args.detector,
        classifier_variant=args.classifier,
        device=device,
        confidence=args.conf,
    )
    publisher = StopPublisher(config.pipeline.get("topic", "stop_signal"), args.no_ros)
    fps_counter = FPSCounter()
    last_stop = None
    writer = None
    processed = 0

    def process(frame):
        nonlocal last_stop, writer, processed
        try:
            result = pipeline.process_frame(frame)
        except Exception:
            LOGGER.exception("Frame %d failed; publishing GO for this frame", processed + 1)
            publisher.publish(0)
            if last_stop is not False:
                LOGGER.warning("GO: frame processing failed")
                last_stop = False
            return True
        publisher.publish(int(result.stop))
        if result.stop != last_stop:
            state = "STOP" if result.stop else "GO"
            reason = ", ".join(result.reasons) if result.reasons else "no trigger objects"
            LOGGER.info("%s: %s", state, reason)
            last_stop = result.stop
        fps = fps_counter.tick()
        annotated = result.image
        draw_status(annotated, result.stop, fps)
        if args.save_video:
            output_path = Path(args.save_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (annotated.shape[1], annotated.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video output: {output_path}")
            writer.write(annotated)
        if not args.no_display:
            cv2.imshow("Realtime stop-signal pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        processed += 1
        return True

    LOGGER.info("Starting source=%s detector=%s classifier=%s device=%s",
                source, args.detector or config.pipeline["models"]["default_detector"],
                args.classifier or config.pipeline["models"]["default_classifier"],
                pipeline.detector.device)
    try:
        if isinstance(source, int):
            process_camera(source, process, args.max_frames)
        elif Path(source).is_dir() or Path(source).suffix.lower() in IMAGE_EXTENSIONS:
            process_images(source, process, args.max_frames)
        elif Path(source).suffix.lower() in VIDEO_EXTENSIONS:
            process_stream(source, process, args.max_frames)
        else:
            raise ValueError(f"Unsupported source: {source}")
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; shutting down")
    finally:
        publisher.publish(0)
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        publisher.close()
    LOGGER.info("Processed %d frame(s)", processed)


if __name__ == "__main__":
    main()
