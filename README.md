# Realtime Road-Object Stop Signal

This project runs the trained YOLO11 detector and sign classifier on camera frames, estimates ranges for stop-triggering objects, annotates the feed, and publishes a ROS2 stop signal. Models are loaded once at startup; this repository's inference pipeline does not retrain them.

## Repository structure

```text
configs/       camera, object-size, detector-role, and pipeline settings
src/           inference modules and realtime pipeline
scripts/       run entry point, image-test utility, and development checks
weights/       trained detector and sign-classifier checkpoints
data/          datasets (local, ignored by git)
results/demo/  annotated image-test outputs and detections.csv
```

## Run the live pipeline

The default source is camera index `0`. The entry point activates the existing ML environment and ROS2 Humble environment in the required order:

```bash
cd /home/marsuser/assignment-eric
./scripts/run_all.sh
```

To select a camera, detector/classifier checkpoint, confidence, or output video:

```bash
cd /home/marsuser/assignment-eric
./scripts/run_all.sh --source 0 --detector y11s --classifier sign_s --conf 0.35
```

`--source` also accepts a video path, image path, or image folder. `--device auto` selects CUDA when available and CPU otherwise. Use `--no-display` for headless execution, `--no-ros` for console-only signal output, `--save-video results/demo/live.mp4` to record annotated frames, or `--max-frames N` for a bounded run. Press `q` to quit the display.

Run an offline test over a small sample of the held-out images:

```bash
cd /home/marsuser/assignment-eric
source /home/marsuser/ml_env/bin/activate
python scripts/run_image_test.py --max-images 6
```

The test writes annotated images and `results/demo/detections.csv`.

## Configuration

Edit [configs/camera.yaml](configs/camera.yaml) with calibrated `fx_px`, `fy_px`, principal point (`cx_px`, `cy_px`), calibration image dimensions, and distortion coefficients. The initial horizontal FOV is only an assumption. If pixel focal length is unavailable, provide both `sensor_width_mm` and `focal_length_mm`; otherwise geometry uses the configured FOV. `depth_scale` optionally rescales model depth.

Edit [configs/object_sizes.yaml](configs/object_sizes.yaml) to replace approximate heights and widths for barriers, cones, traffic lights, and sign types. Geometry uses the configured height by default.

Edit `HSV_RANGES`, `SATURATION_MIN`, `VALUE_MIN`, and `MIN_COLOR_PIXEL_FRACTION` at the top of [src/traffic_light.py](src/traffic_light.py) to tune traffic-light color segmentation for the camera.

[configs/pipeline.yaml](configs/pipeline.yaml) selects the trained detector/classifier weights, detector class-to-role mapping, default variants, thresholds, device, camera source, and ROS topic. The detector names verified from the trained weights are `barrier`, `cone`, `signBoard`, and `trafficLight`; the classifier outputs `barrier_ahead`, `other`, and `stop`.

## Stop logic and ROS2

Each processed frame publishes `std_msgs/Int32` to `stop_signal` (`/stop_signal` when viewed as a ROS graph name). The value is `1` if the frame contains any detected barrier, a red traffic light, a sign classified as `stop`, or a sign classified as `barrier ahead`; otherwise it is `0`. A message is published for every processed frame. Depth inference is skipped on frames with no trigger, and runs once per trigger frame. Trigger objects are reported with geometry and depth distances when both are available.

Observe the topic from another terminal after sourcing ROS2:

```bash
source /opt/ros/humble/setup.bash
timeout 10 ros2 topic echo /stop_signal
timeout 10 ros2 topic hz /stop_signal
```

## TODO

- Edge optimization (quantization/pruning/FPS benchmark): not done yet.
