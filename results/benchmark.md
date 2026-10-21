# Edge optimization benchmark

Hardware: **CPU:** AMD Ryzen 9 5950X 16-Core Processor; **GPU:** NVIDIA RTX A4000.

Sample: 50 fixed test images; 10 warmup; 40 timed inferences; imgsz=640, batch=1, rect=False.

FP16 uses Ultralytics `quantize=16`, the supported replacement for deprecated `half=True`.

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
