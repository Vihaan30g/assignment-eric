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

## Edge Optimization

Hardware: **CPU:** AMD Ryzen 9 5950X 16-Core Processor; **GPU:** NVIDIA RTX A4000.

Fixed sample: 50 images; 10 warmup inferences; 40 timed inferences per configuration with fixed 640x640 input. Inference uses `rect=False`. Validation: `data/assignment1_clean/data.yaml` with `split=test`, `imgsz=640`, `batch=1`.

On this Ultralytics 8.4 installation, FP16 uses `quantize=16`, the supported replacement for the deprecated `half=True` argument.

| Configuration | Device | Precision | Mean latency (ms) | FPS | mAP50 | Status / notes |
| --- | --- | --- | --- | --- | --- | --- |
| y11s GPU FP32 | cuda:0 | FP32 | 5.9785 | 167.2674 | 0.965556 | ok |
| y11s CPU FP32 | cpu | FP32 | 61.9163 | 16.1508 | 0.965577 | ok |
| y11n GPU FP32 | cuda:0 | FP32 | 5.9626 | 167.7121 | 0.958561 | ok |
| y11n CPU FP32 | cpu | FP32 | 25.4075 | 39.3584 | 0.958549 | ok |
| y11n GPU FP16 | cuda:0 | FP16 | 5.8234 | 171.7221 | 0.958893 | ok |
| y11n ONNX CPU | CPU | ONNX FP32 | 70.4948 | 14.1855 | — | mAP50 skipped; benchmarked with ONNX Runtime 1.23.2. |
| y11n INT8 | N/A | INT8 | — | — | — | INT8: skipped; a quantization runtime is present, but this benchmark does not assume a calibrated INT8 export path or install/configure one. |

### Before vs after

Before: y11s GPU FP32 measured 167.27 FPS at mAP50 0.9656.
After: y11n GPU FP16 measured 171.72 FPS (1.03x) at mAP50 0.9589.
Trade-off: FP16 y11n mAP50 changed by -0.0067; y11n CPU FP32 was 39.3584 FPS.

ONNX: exported to `weights/export/y11n_best.onnx` and benchmarked on CPU. INT8: skipped; a quantization runtime is present, but this benchmark does not assume a calibrated INT8 export path or install/configure one.

## Extra Credit

The reference inputs are `data/calib/homography_ref.jpg` and `data/videos/flow_test.mp4`. Demo outputs are saved in `results/extras/`.

### Ground-plane homography

Calibration is still pending the four real corner clicks; no pixel points or bird's-eye result have been fabricated. Run:

```bash
cd ~/assignment-eric && source /home/marsuser/ml_env/bin/activate && python -m src.extras.birdseye --select-points
```

Click top-left, top-right, bottom-right, then bottom-left (clockwise from the far edge); press `r` to reset and `s` to save. The tool checks corner reprojection and then writes `results/extras/birdseye_real.png` with a 10 cm metric grid and detector markers.

The camera is uncalibrated and lens distortion is ignored. This mapping applies only to the flat plane containing the sheet; accuracy decreases away from the sheet, and the camera must not move after calibration.

### Global-motion-compensated optical flow

![Real optical-flow representative frame](results/extras/optical_flow_real.png)

Run automatic detection every third frame across the full supplied clip:

```bash
cd ~/assignment-eric && source /home/marsuser/ml_env/bin/activate && python -m src.extras.optical_flow --source data/videos/flow_test.mp4 --no-display
```

This run found the cone at frame 36 (object ID 1) in automatic mode. The annotated [video](results/extras/optical_flow_real.mp4) and per-frame velocity [CSV](results/extras/optical_flow_real.csv) accompany the representative frame above. If automatic detection fails on another clip, `--roi x,y,w,h --start-frame N` or `--select-roi --start-frame N` starts manual tracking at frame `N`.

Lucas–Kanade object flow is corrected by median background flow; `MOVING` uses the configured px/s threshold. This is image-plane velocity, not world velocity, and assumes a mostly static camera; camera motion is only approximately removed.

### Stereo depth

[docs/epipolar_derivation.md](docs/epipolar_derivation.md) derives `Z = fB/d`; [src/extras/epipolar.py](src/extras/epipolar.py) provides disparity/depth conversion and a StereoSGBM demo. No real left/right stereo pair was supplied, so no disparity output was generated.
