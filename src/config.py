"""Configuration loading and validation for the realtime pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return result


@dataclass(frozen=True)
class PipelineConfig:
    pipeline: dict[str, Any]
    camera: dict[str, Any]
    object_sizes: dict[str, Any]

    @property
    def roles(self) -> dict[str, str]:
        return self.pipeline["class_roles"]


def load_config(
    pipeline_path: str | Path = ROOT / "configs/pipeline.yaml",
    camera_path: str | Path = ROOT / "configs/camera.yaml",
    object_sizes_path: str | Path = ROOT / "configs/object_sizes.yaml",
) -> PipelineConfig:
    pipeline_data = load_yaml(pipeline_path)
    camera_data = load_yaml(camera_path)
    sizes_data = load_yaml(object_sizes_path)
    required = ("models", "detection", "class_roles", "sign_labels")
    missing = [section for section in required if section not in pipeline_data]
    if missing:
        raise ValueError(f"Missing pipeline config section(s): {', '.join(missing)}")
    if "camera" not in camera_data or "objects" not in sizes_data:
        raise ValueError("camera.yaml needs a camera mapping and object_sizes.yaml needs objects")
    return PipelineConfig(
        pipeline=pipeline_data,
        camera=camera_data["camera"],
        object_sizes=sizes_data["objects"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = load_config()
    print(f"Loaded {len(config.roles)} detector roles and {len(config.object_sizes)} object sizes")
