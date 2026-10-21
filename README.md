# Real-Time Road-Object Detection, Depth Estimation, and Stop Signaling

**Author:** Vihaan Gupta

## Abstract

I chose this mini project because I had not previously explored how a model
estimates depth from RGB image data. I used the project as an opportunity to
study monocular depth estimation alongside object detection and to connect
those results to a live robotics-style response. The application combines
trained object-detection and sign-classification models with a UniDepth V2
monocular depth model, geometry-based distance estimation, and ROS2 stop
signaling. The models are used for inference; the runtime application does not
retrain them.

The live workflow detects barriers, cones, sign boards, and traffic lights.
It classifies sign-board crops and traffic-light color, estimates distance for
objects, annotates the camera view, and publishes an integer stop state. Depth
inference is run only when a frame contains a stop-triggering object.

**Demo video:** [Watch the recorded project demo](https://youtu.be/lSo4V8ctqcc).

## Contents

- [Project overview](#project-overview)
- [Repository structure](#repository-structure)
- [Implemented code and models](#implemented-code-and-models)
- [End-to-end workflow](#end-to-end-workflow)
- [Run the project](#run-the-project)
- [Distance estimation](#distance-estimation)
- [Stop logic and ROS2 topic](#stop-logic-and-ros2-topic)
- [Edge optimization results](#edge-optimization-results)
- [Extra-credit modules](#extra-credit-modules)
- [Demo video](#demo-video)
- [Limitations and TODO](#limitations-and-todo)
- [Contact](#contact)

## Project overview

The runtime detector class names are `barrier`, `cone`, `signBoard`, and
`trafficLight`. The trained sign classifier outputs `barrier_ahead`, `other`,
and `stop`. The configured defaults are y11s for detection, sign_s for sign
classification, and UniDepth V2 `vits14` for monocular depth.

There are two distance paths:

1. **Geometry distance:** uses an object's configured physical size, its
   bounding-box size in pixels, and an approximate camera focal length.
2. **Monocular depth:** UniDepth V2 predicts a depth map from the RGB frame.
   On trigger frames, the system reduces depth values within the detected
   object's central box region to one displayed range estimate.

These paths provide image-based range estimates, but neither replaces
calibration or independent range validation. Details and constraints are in
[Distance estimation](#distance-estimation) and
[Limitations and TODO](#limitations-and-todo).

## Repository structure

```text
configs/
  camera.yaml             Camera intrinsics/FOV, depth scale, stereo baseline
  object_sizes.yaml       Approximate physical dimensions for geometry range
  pipeline.yaml           Model paths, roles, thresholds, device, ROS topic
  homography.yaml         Ground-plane calibration settings and saved points
  optical_flow.yaml       Optical-flow threshold and detector settings
src/
  detect.py               YOLO detector wrapper and typed detections
  sign_classifier.py      Classifies detected sign-board crops
  traffic_light.py        HSV-based red/yellow/green light color estimate
  distance_geometry.py    Pinhole-model range from known object dimensions
  distance_depth.py       UniDepth V2 depth-map inference and box reduction
  pipeline.py             Per-frame detection/classification/range/trigger flow
  ros_publisher.py        std_msgs/Int32 stop-state publisher
  run_realtime.py         Camera/video/image input loop and display/output
  extras/
    birdseye.py           Click-calibrated ground-plane homography / BEV
    optical_flow.py       LK object flow with background-motion subtraction
    epipolar.py           Stereo disparity-depth conversions and SGBM demo
scripts/
  run_all.sh              Activates the existing ML and ROS2 environments
  run_image_test.py       Runs the detector on held-out images
  benchmark.py            Measures latency/FPS and detector mAP50
weights/                   Trained detector/classifier checkpoints and ONNX
data/                      Local datasets and real extra-credit inputs
results/                   Image tests, benchmarks, and extra-credit outputs
docs/                      Stereo epipolar-geometry derivation
```

Datasets and model files are local project assets. The detector's trained class
IDs map to semantic roles through [configs/pipeline.yaml](configs/pipeline.yaml).

## Implemented code and models

### Main real-time application

- **Detection:** [src/detect.py](src/detect.py) wraps the trained Ultralytics
  YOLO detector. It returns bounding boxes, confidence, class name, and the
  configured semantic role (`barrier`, `cone`, `sign_board`, or
  `traffic_light`).
- **Sign classification:** [src/sign_classifier.py](src/sign_classifier.py)
  classifies each detected sign-board crop using the configured trained
  classifier and maps its output to `stop`, `barrier ahead`, or `others`.
- **Traffic-light color:** [src/traffic_light.py](src/traffic_light.py)
  analyzes the detected light crop in HSV and returns a color estimate and
  confidence. It is a color-segmentation step, not a separate learned
  traffic-light-state model.
- **Distance estimates:** [src/distance_geometry.py](src/distance_geometry.py)
  implements the pinhole formula using known object dimensions, and
  [src/distance_depth.py](src/distance_depth.py) runs UniDepth V2 and extracts
  a robust central-region depth estimate for a detected box.
- **Frame orchestration:** [src/pipeline.py](src/pipeline.py) coordinates
  detection, per-role classification, trigger decisions, optional depth
  inference, distance labels, and annotated output.
- **Input/display loop:** [src/run_realtime.py](src/run_realtime.py) accepts
  camera, video, image, and image-folder input. For a live camera, a reader
  thread keeps the newest frame available so inference does not intentionally
  build up a queue of stale frames.
- **ROS2 output:** [src/ros_publisher.py](src/ros_publisher.py) publishes the
  stop state as `std_msgs/Int32` and logs state changes if ROS2 publishing is
  disabled or unavailable.

### Extra-credit code

- [src/extras/birdseye.py](src/extras/birdseye.py) provides a four-point
  click-to-calibrate homography and a metric bird's-eye view with a grid and
  projected detection points. Real calibration clicks are still pending.
- [src/extras/optical_flow.py](src/extras/optical_flow.py) tracks detected
  cone/barrier features with Lucas–Kanade optical flow, subtracts median
  background flow, and writes annotated video, a preview frame, and per-frame
  velocity CSV. The supplied video was processed in automatic mode.
- [src/extras/epipolar.py](src/extras/epipolar.py) contains stereo
  disparity/depth conversion functions and an OpenCV StereoSGBM demo.
  [docs/epipolar_derivation.md](docs/epipolar_derivation.md) derives the
  rectified-stereo depth relation. No real stereo pair has been supplied.

## End-to-end workflow

The live demo sequence is:

1. **Start the application.** The launcher activates the existing ML
   environment and ROS2 Humble, then starts the configured pipeline. The
   detector, sign classifier, and depth model are loaded for inference.
2. **Read a camera frame.** The default input is camera index `0`. A camera
   reader keeps the latest frame; files can also be processed in sequence.
3. **Detect objects.** The detector finds boxes and assigns class names and
   roles for barriers, cones, sign boards, and traffic lights.
4. **Analyze relevant detections.** A sign-board crop is sent to the sign
   classifier. A traffic-light crop is evaluated by the HSV color routine.
   Cones and non-trigger objects are still detected and can receive geometry
   range labels.
5. **Decide whether the frame requires STOP.** A barrier, a red traffic light,
   a `stop` sign, or a `barrier ahead` sign sets the frame's stop state to `1`.
   If none is present, the frame state is `0`.
6. **Estimate and annotate distance.** Geometry distance is computed from
   configured size and bounding-box dimensions. On trigger frames, UniDepth
   V2 infers a depth map from the RGB frame; the median of valid depth pixels
   in the central part of the trigger object's box is used as its depth
   estimate. Available estimates are drawn with the object label.
7. **Publish and display.** The state is published for every processed frame
   to `/stop_signal`. The application overlays the STOP/GO state and frame
   rate, displays the annotated frame unless headless mode is selected, and
   can optionally record the annotated stream.
8. **Observe the ROS2 result.** In a second ROS2-sourced terminal, inspect
   `/stop_signal`. The demo recording is intended to show the camera view,
   objects entering the view, a stop sign triggering depth inference, and
   value `1` appearing on the ROS2 topic.

If processing a frame raises an exception in the realtime loop, the current
implementation logs the failure and publishes GO (`0`) for that frame rather
than leaving the previously published STOP state latched.

## Run the project

From the repository root, launch the default live camera pipeline:

```bash
./scripts/run_all.sh
```

The script sources `/home/marsuser/ml_env` and `/opt/ros/humble` before
launching. To choose a source and model variants:

```bash
./scripts/run_all.sh --source 0 --detector y11s --classifier sign_s --conf 0.35
```

`--source` accepts a camera index, video path, image path, or image-folder path.
`--device auto` selects CUDA if available and CPU otherwise. Useful options:

```text
--no-display                 Run without an OpenCV display window
--no-ros                     Disable ROS2 publishing (console fallback)
--save-video PATH            Save annotated output video
--max-frames N               Stop after N processed frames
```

Run a small held-out image test or the saved edge benchmark with the existing
environment (no dependency installation is needed for these commands):

```bash
source /home/marsuser/ml_env/bin/activate
python scripts/run_image_test.py --max-images 6
python scripts/benchmark.py
```

The image test writes annotated frames and
`results/demo/detections.csv`. Benchmark files are
`results/benchmark.csv`, `results/benchmark.md`, and
`results/fps_comparison.png`.

## Distance estimation

### Geometry: pinhole-camera formula

For an object with known physical height, the apparent size decreases with
distance. Under the pinhole-camera model:

```text
Z = f * H / h
```

where:

- `Z` is the estimated distance in metres,
- `f` is the focal length in pixels along the measured image axis,
- `H` is the configured physical object height in metres, and
- `h` is the detected bounding-box height in pixels.

The implementation uses the box width and configured physical width if width
is selected instead. Focal length is read from calibrated `fx_px`/`fy_px`,
derived from sensor and lens dimensions, or estimated from horizontal field of
view. The current camera file uses a 70-degree FOV assumption, not measured
intrinsics. Physical dimensions are approximate values in
[configs/object_sizes.yaml](configs/object_sizes.yaml), so geometry output is
an estimate, not a calibrated measurement.

### Monocular depth: UniDepth V2

The depth model receives the RGB-converted camera frame and predicts a dense
depth map. When the camera configuration provides usable parameters, the
pipeline also supplies an intrinsic matrix; currently its focal length is
based on the uncalibrated FOV assumption. For a trigger object's box, the code
takes the median of finite, positive depth samples from the central half-width
and half-height region. This reduces the influence of box-edge background
pixels but cannot guarantee that the box contains only the object.

Geometry and model depth are shown together when both are available. The
project does not yet claim quantitatively validated depth accuracy against
ground-truth ranges. Do not interpret a plausible-looking color/depth estimate
as evidence of calibration.

## Stop logic and ROS2 topic

The configured topic is `stop_signal`, shown by ROS2 tools as `/stop_signal`.
The published type is `std_msgs/Int32`:

| Published value | Meaning |
| ---: | --- |
| `1` | At least one stop-triggering object is present in this frame |
| `0` | No stop-triggering object is present in this frame |

The trigger rules are:

- any detected **barrier**;
- a **traffic light** whose HSV color estimate is red;
- a **sign board** classified as `stop`; or
- a **sign board** classified as `barrier ahead`.

The publisher is called for each processed frame. Depth inference is skipped
on frames with no trigger and performed once on a trigger frame; it is not
used to determine the Boolean stop decision.

To inspect messages and publication rate, open another terminal:

```bash
source /opt/ros/humble/setup.bash
timeout 10 ros2 topic echo /stop_signal
timeout 10 ros2 topic hz /stop_signal
```

If ROS2 Python message packages are unavailable or `--no-ros` is passed, the
publisher uses a console fallback instead.

## Edge optimization results

Hardware used: **CPU:** AMD Ryzen 9 5950X 16-Core Processor; **GPU:** NVIDIA
RTX A4000. The detector benchmark used 50 fixed test images, 10 warmup runs,
and 40 timed runs per configuration at `imgsz=640`, `batch=1`, `rect=False`.
mAP50 is reported on the configured test split. These detector benchmark
numbers are not end-to-end camera-pipeline FPS.

| Configuration | Device | Precision | Mean latency (ms) | FPS | mAP50 | Status / notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| y11s GPU | cuda:0 | FP32 | 5.9785 | 167.2674 | 0.965556 | measured |
| y11s CPU | cpu | FP32 | 61.9163 | 16.1508 | 0.965577 | measured |
| y11n GPU | cuda:0 | FP32 | 5.9626 | 167.7121 | 0.958561 | measured |
| y11n CPU | cpu | FP32 | 25.4075 | 39.3584 | 0.958549 | measured |
| y11n GPU | cuda:0 | FP16 | 5.8234 | 171.7221 | 0.958893 | measured; `quantize=16` |
| y11n ONNX | CPU | FP32 | 70.4948 | 14.1855 | — | ONNX Runtime; mAP50 not measured |
| y11n INT8 | — | INT8 | — | — | — | skipped; calibrated INT8 export path not configured |

Before optimization, y11s GPU FP32 measured 167.27 FPS at mAP50 0.9656.
Afterward, y11n GPU FP16 measured 171.72 FPS (about 1.03×) at mAP50 0.9589.
That is a small speed increase, with a 0.0067 lower mAP50; y11n CPU FP32
measured 39.36 FPS. This comparison covers model/precision configurations; it
does **not** mean the models were pruned.

## Extra-credit modules

### Ground-plane bird's-eye view

The real reference image is `data/calib/homography_ref.jpg`. Calibration has
not been completed: the required four corners must be clicked by the user, so
no real bird's-eye result is claimed. Start the calibration UI with:

```bash
python -m src.extras.birdseye --select-points
```

Select top-left, top-right, bottom-right, and bottom-left, clockwise from the
far edge. Press `r` to reset and `s` to save. Once calibrated, the output is
`results/extras/birdseye_real.png`. The transform is valid only on the flat
plane of the reference rectangle; lens distortion is ignored, accuracy falls
away from the plane, and the camera must not move.

### Global-motion-compensated optical flow

![Optical-flow representative frame](results/extras/optical_flow_real.png)

The real clip `data/videos/flow_test.mp4` was processed in automatic mode,
checking the detector every third frame. A cone was detected at frame 36.
Lucas–Kanade features track object motion; median background flow is subtracted
to approximately compensate for camera motion. Output files are
`results/extras/optical_flow_real.mp4`,
`results/extras/optical_flow_real.png`, and
`results/extras/optical_flow_real.csv`.

```bash
python -m src.extras.optical_flow --source data/videos/flow_test.mp4 --no-display
```

The CSV reports image-plane velocity in pixels per second and a
`MOVING`/`STATIC` flag using the configured threshold. This is not world-space
velocity; camera movement can remain in the result. For a clip where detection
does not find an object, manual tracking is available with
`--roi x,y,w,h --start-frame N` or interactive
`--select-roi --start-frame N`.

### Stereo disparity and depth

[src/extras/epipolar.py](src/extras/epipolar.py) implements disparity/depth
conversion and a StereoSGBM example for a left/right image pair.
[docs/epipolar_derivation.md](docs/epipolar_derivation.md) derives `Z = fB/d`
for a rectified pair. No real stereo image pair was supplied, so no stereo
output is included. Stereo depth requires calibrated and rectified cameras.

## Demo video

**YouTube demo placeholder — upload link not available yet.**

[▶ Watch the full project demo video on YouTube](https://youtu.be/lSo4V8ctqcc)

The video demonstrates the live run, including objects in the camera view, a
stop sign, the resulting depth estimate, and the ROS2 `/stop_signal` output.

## Limitations and TODO

- **Pruning is not done.** No trained weights have been structurally or
  sparsely pruned. The benchmark only compares existing detector variants,
  FP16, and an ONNX export; INT8 was skipped because a calibrated quantization
  path was not configured. No model was retrained as part of the runtime work.
- **Camera calibration is not done.** `configs/camera.yaml` has placeholder
  intrinsics and uses a 70-degree horizontal-FOV assumption. Lens distortion
  is not calibrated, and the homography corner-selection step is pending.
- **Depth accuracy is not yet ground-truth validated.** UniDepth provides a
  monocular estimate from RGB; the supplied camera intrinsics are not measured.
- **Object dimensions are approximate.** Replace values in
  `configs/object_sizes.yaml` with measurements for the actual scene.
- **Stereo depth is not demonstrated on real stereo data.** A calibrated,
  rectified left/right pair and measured baseline are required.
- **Homography is plane-specific.** It does not map arbitrary 3D points and
  requires a stationary camera.
- **Optical-flow motion is approximate.** Background compensation does not
  remove all camera motion; reported speed is in pixels per second.
- **Full-pipeline performance differs from detector benchmark speed.** Sign
  classification, triggered depth inference, image display, and ROS2 publishing
  add work that is not included in the detector-only FPS table.

## Contact

- **Name:** Vihaan Gupta
- **Mobile:** +91 9399929308
- **Email:** [vihaan30g@gmail.com](mailto:vihaan30g@gmail.com)
- **Portfolio:** [vihaan30g.github.io](https://vihaan30g.github.io)
