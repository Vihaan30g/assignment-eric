#!/usr/bin/env python3
"""Benchmark trained YOLO11 detectors on a fixed held-out image sample."""
import argparse
import csv
import importlib.util
import math
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "data/assignment1_clean/test/images"
DATA_YAML = ROOT / "data/assignment1_clean/data.yaml"
RESULTS = ROOT / "results"
CSV_PATH = RESULTS / "benchmark.csv"
MARKDOWN_PATH = RESULTS / "benchmark.md"
CHART_PATH = RESULTS / "fps_comparison.png"
IMAGE_COUNT = 50
WARMUP_COUNT = 10
IMAGE_SIZE = 640
BATCH_SIZE = 1
SAMPLE_SEED = 20261021


def cpu_model_name():
    processor = platform.processor().strip()
    if processor and processor.lower() not in {"x86_64", "amd64", "aarch64", "arm64"}:
        return processor
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def collect_images():
    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(f"Test image directory does not exist: {TEST_IMAGES}")
    paths = sorted(
        path for path in TEST_IMAGES.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(paths) < IMAGE_COUNT:
        raise ValueError(f"Need at least {IMAGE_COUNT} test images, found {len(paths)}")
    random.Random(SAMPLE_SEED).shuffle(paths)
    selected = paths[:IMAGE_COUNT]
    images = []
    for path in selected:
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"OpenCV could not read benchmark image: {path}")
        images.append(image)
    return images


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark_latency(model, images, device, half=False):
    # Ultralytics 8.4 replaces the deprecated half flag with quantize=16.
    quantize = 16 if half else None
    for index in range(WARMUP_COUNT):
        model.predict(
            images[index],
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            rect=False,
            device=device,
            quantize=quantize,
            verbose=False,
        )
    synchronize(device)

    latencies_ms = []
    for image in images[WARMUP_COUNT:]:
        synchronize(device)
        start = time.perf_counter()
        model.predict(
            image,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            rect=False,
            device=device,
            quantize=quantize,
            verbose=False,
        )
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    return mean_latency_ms, 1000.0 / mean_latency_ms


def evaluate_map50(model, variant, device, half, validation_root):
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        quantize=16 if half else None,
        plots=False,
        verbose=False,
        cache=False,
        project=str(validation_root),
        name=variant,
        exist_ok=True,
    )
    return float(metrics.box.map50)


def make_row(config, model_name, device, precision, latency, fps, map50, status, notes=""):
    return {
        "config": config,
        "model": model_name,
        "device": device,
        "precision": precision,
        "mean_latency_ms": "" if latency is None else f"{latency:.4f}",
        "fps": "" if fps is None else f"{fps:.4f}",
        "mAP50": "" if map50 is None else f"{map50:.6f}",
        "status": status,
        "notes": notes,
    }


def run_variant(
    rows,
    images,
    weights_path,
    config_name,
    model_name,
    device,
    precision,
    half,
    validation_root,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        note = "Skipped: CUDA is unavailable."
        print(f"{config_name}: {note}")
        rows.append(make_row(config_name, model_name, device, precision, None, None, None, "skipped", note))
        return

    print(f"\n== {config_name}: imgsz={IMAGE_SIZE}, batch={BATCH_SIZE}, precision={precision} ==")
    model = YOLO(str(weights_path))
    mean_latency_ms, fps = benchmark_latency(model, images, device, half=half)
    print(f"Mean latency: {mean_latency_ms:.3f} ms; FPS: {fps:.2f}")
    map50 = evaluate_map50(
        model,
        config_name.replace("/", "_").replace(" ", "_"),
        device,
        half,
        validation_root,
    )
    print(f"mAP50: {map50:.6f}")
    rows.append(
        make_row(
            config_name, model_name, device, precision,
            mean_latency_ms, fps, map50, "ok",
        )
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_onnx(rows, images):
    dependencies = ("onnx", "onnxruntime")
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        note = f"Skipped: not installed ({', '.join(missing)}); no packages were installed."
        print(f"y11n ONNX CPU: {note}")
        rows.append(make_row("y11n ONNX CPU", "y11n", "CPU", "FP32", None, None, None, "skipped", note))
        return None

    export_dir = ROOT / "weights/export"
    export_dir.mkdir(parents=True, exist_ok=True)
    import onnxruntime

    model = YOLO(str(ROOT / "weights/y11n_best.pt"))
    exported = model.export(format="onnx", imgsz=IMAGE_SIZE)
    exported_path = Path(exported)
    if not exported_path.is_file():
        raise FileNotFoundError(f"Ultralytics did not create an ONNX file at {exported_path}")
    onnx_path = export_dir / "y11n_best.onnx"
    if exported_path.resolve() != onnx_path.resolve():
        shutil.copy2(exported_path, onnx_path)
    del model

    print(f"ONNX Runtime: {onnxruntime.__version__}; model: {onnx_path}")
    onnx_model = YOLO(str(onnx_path), task="detect")
    mean_latency_ms, fps = benchmark_latency(onnx_model, images, "cpu")
    print(f"y11n ONNX CPU mean latency: {mean_latency_ms:.3f} ms; FPS: {fps:.2f}")
    rows.append(
        make_row(
            "y11n ONNX CPU", "y11n", "CPU", "ONNX FP32",
            mean_latency_ms, fps, None, "ok",
            f"mAP50 skipped; benchmarked with ONNX Runtime {onnxruntime.__version__}.",
        )
    )
    del onnx_model
    return onnx_path


def write_csv(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "config", "model", "device", "precision", "mean_latency_ms",
                "fps", "mAP50", "status", "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def table_for(rows):
    columns = (
        ("Configuration", "config"),
        ("Device", "device"),
        ("Precision", "precision"),
        ("Mean latency (ms)", "mean_latency_ms"),
        ("FPS", "fps"),
        ("mAP50", "mAP50"),
        ("Status / notes", "notes"),
    )
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key] or ("—" if key != "notes" else row["status"])
            values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analysis_lines(rows):
    by_config = {row["config"]: row for row in rows}
    baseline = by_config.get("y11s GPU FP32")
    optimized = by_config.get("y11n GPU FP16")
    cpu_row = by_config.get("y11n CPU FP32")
    lines = []
    if baseline and optimized and baseline["status"] == optimized["status"] == "ok":
        base_fps = float(baseline["fps"])
        optimized_fps = float(optimized["fps"])
        base_map = float(baseline["mAP50"])
        optimized_map = float(optimized["mAP50"])
        speedup = optimized_fps / base_fps if base_fps else math.nan
        lines.append(
            f"Before: y11s GPU FP32 measured {base_fps:.2f} FPS at mAP50 {base_map:.4f}."
        )
        lines.append(
            f"After: y11n GPU FP16 measured {optimized_fps:.2f} FPS "
            f"({speedup:.2f}x) at mAP50 {optimized_map:.4f}."
        )
        lines.append(
            f"Trade-off: FP16 y11n mAP50 changed by {optimized_map - base_map:+.4f}; "
            f"y11n CPU FP32 was {cpu_row['fps']} FPS."
            if cpu_row and cpu_row["status"] == "ok"
            else f"Trade-off: FP16 y11n mAP50 changed by {optimized_map - base_map:+.4f}."
        )
    else:
        lines = [
            "Before: y11s GPU FP32 baseline was unavailable.",
            "After: y11n GPU FP16 result was unavailable.",
            "Trade-off: see the status and notes in the benchmark table.",
        ]
    return lines


def update_readme(rows, machine_text, skipped_notes):
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    table = table_for(rows)
    analysis = analysis_lines(rows)
    section = (
        "## Edge Optimization\n\n"
        f"{machine_text}\n\n"
        f"Fixed sample: {IMAGE_COUNT} images; {WARMUP_COUNT} warmup inferences; "
        f"{IMAGE_COUNT - WARMUP_COUNT} timed inferences per configuration with fixed 640x640 input. "
        f"Inference uses `rect=False`. Validation: `data/assignment1_clean/data.yaml` with `split=test`, "
        f"`imgsz={IMAGE_SIZE}`, `batch={BATCH_SIZE}`.\n\n"
        "On this Ultralytics 8.4 installation, FP16 uses `quantize=16`, the supported replacement "
        "for the deprecated `half=True` argument.\n\n"
        f"{table}\n\n"
        "### Before vs after\n\n"
        + "\n".join(analysis)
        + "\n\n"
        + skipped_notes
        + "\n"
    )

    start = readme.find("## Edge Optimization")
    if start >= 0:
        next_heading = readme.find("\n## ", start + 1)
        readme = readme[:start] + section + (
            "\n" + readme[next_heading + 1:] if next_heading >= 0 else ""
        )
    else:
        legacy_todo = "## TODO\n\n- Edge optimization (quantization/pruning/FPS benchmark): not done yet.\n"
        if legacy_todo in readme:
            readme = readme.replace(legacy_todo, section, 1)
        else:
            readme = readme.rstrip() + "\n\n" + section
    readme_path.write_text(readme, encoding="utf-8")


def write_chart(rows, cpu_name, gpu_name):
    plotted = [row for row in rows if row["status"] == "ok" and row["fps"]]
    figure, axis = plt.subplots(figsize=(11, 6))
    labels = [row["config"] for row in plotted]
    values = [float(row["fps"]) for row in plotted]
    bars = axis.bar(labels, values, color="#3478b8")
    axis.set_ylabel("Frames per second")
    axis.set_title(f"YOLO11 detector speed (CPU: {cpu_name}; GPU: {gpu_name})")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    axis.bar_label(bars, fmt="%.1f", padding=3)
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    cpu_name = cpu_model_name()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable"
    print(f"CPU model: {cpu_name}")
    print(f"GPU model: {gpu_name}")

    images = collect_images()
    print(
        f"Fixed sample: {len(images)} images (seed {SAMPLE_SEED}); "
        f"{WARMUP_COUNT} warmup and {len(images) - WARMUP_COUNT} timed runs; "
        f"imgsz={IMAGE_SIZE}, batch={BATCH_SIZE}."
    )

    rows = []
    validation_cache = DATA_YAML.parent / "test/labels.cache"
    cache_existed_before_run = validation_cache.exists()
    try:
        with tempfile.TemporaryDirectory(prefix="assignment-eric-benchmark-") as temp_dir:
            validation_root = Path(temp_dir) / "validation"
            run_variant(
                rows, images, ROOT / "weights/y11s_best.pt",
                "y11s GPU FP32", "y11s", "cuda:0", "FP32", False, validation_root,
            )
            run_variant(
                rows, images, ROOT / "weights/y11s_best.pt",
                "y11s CPU FP32", "y11s", "cpu", "FP32", False, validation_root,
            )
            run_variant(
                rows, images, ROOT / "weights/y11n_best.pt",
                "y11n GPU FP32", "y11n", "cuda:0", "FP32", False, validation_root,
            )
            run_variant(
                rows, images, ROOT / "weights/y11n_best.pt",
                "y11n CPU FP32", "y11n", "cpu", "FP32", False, validation_root,
            )
            run_variant(
                rows, images, ROOT / "weights/y11n_best.pt",
                "y11n GPU FP16", "y11n", "cuda:0", "FP16", True, validation_root,
            )
        onnx_path = run_onnx(rows, images)
    finally:
        if not cache_existed_before_run and validation_cache.is_file():
            validation_cache.unlink()
    if onnx_path is None:
        onnx_note = "ONNX: skipped; onnx and/or onnxruntime is not installed."
    else:
        onnx_note = f"ONNX: exported to `{onnx_path.relative_to(ROOT)}` and benchmarked on CPU."

    quantization_runtimes = ("openvino", "neural_compressor", "onnxruntime", "tensorflow", "tflite_runtime")
    available_quantizers = [
        name for name in quantization_runtimes
        if importlib.util.find_spec(name) is not None
    ]
    if available_quantizers:
        int8_note = (
            "INT8: skipped; a quantization runtime is present, but this benchmark does not "
            "assume a calibrated INT8 export path or install/configure one."
        )
    else:
        int8_note = "INT8: skipped; no supported quantization runtime is installed."
    rows.append(
        make_row(
            "y11n INT8", "y11n", "N/A", "INT8", None, None, None,
            "skipped", int8_note,
        )
    )
    skip_text = f"{onnx_note} {int8_note}"

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    machine_text = f"Hardware: **CPU:** {cpu_name}; **GPU:** {gpu_name}."
    analysis = analysis_lines(rows)
    MARKDOWN_PATH.write_text(
        "# Edge optimization benchmark\n\n"
        f"{machine_text}\n\n"
        f"Sample: {IMAGE_COUNT} fixed test images; {WARMUP_COUNT} warmup; "
        f"{IMAGE_COUNT - WARMUP_COUNT} timed inferences; imgsz={IMAGE_SIZE}, batch={BATCH_SIZE}, "
        "rect=False.\n\n"
        "FP16 uses Ultralytics `quantize=16`, the supported replacement for deprecated `half=True`.\n\n"
        f"{table_for(rows)}\n\n"
        "### Before vs after\n\n"
        + "\n".join(analysis)
        + "\n\n"
        + skip_text
        + "\n",
        encoding="utf-8",
    )
    write_chart(rows, cpu_name, gpu_name)
    update_readme(rows, machine_text, skip_text)

    print(f"\nSaved {CSV_PATH.relative_to(ROOT)}, {MARKDOWN_PATH.relative_to(ROOT)}, "
          f"and {CHART_PATH.relative_to(ROOT)}")
    print("Before vs after:")
    for line in analysis:
        print(f"- {line}")
    print(skip_text)


if __name__ == "__main__":
    main()
