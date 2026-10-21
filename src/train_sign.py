"""Train the 3-class sign classifier (YOLO11-cls) on data/signs_cls.
  python src/train_sign.py --model yolo11n-cls.pt --name sign_n
  python src/train_sign.py --resume --name sign_n
"""
import argparse, json, shutil
from pathlib import Path
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n-cls.pt")
    p.add_argument("--name", default="sign_cls")
    p.add_argument("--data", default=str(ROOT / "data/signs_cls"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    settings.update({"tensorboard": True})
    if a.resume:
        model = YOLO(str(ROOT / "runs" / a.name / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        local = ROOT / "pretrained" / a.model
        model = YOLO(str(local) if local.exists() else a.model)
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                    patience=a.patience, seed=a.seed, plots=True, save_period=5,
                    fliplr=0.0,                       # no mirroring: text and arrows on signs
                    project=str(ROOT / "runs"), name=a.name)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    net = YOLO(str(best))
    m = net.val(data=a.data, split="test", imgsz=a.imgsz,
                project=str(ROOT / "runs"), name=a.name + "_test")
    summary = {"model": a.model, "name": a.name, "top1": float(m.top1),
               "classes": list(net.names.values())}

    out = ROOT / "results" / "training" / a.name
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.png", "*.yaml"):
        for f in save_dir.glob(pattern):
            shutil.copy(f, out / f.name)
    (out / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "weights" / f"{a.name}_best.pt")
    print("Done.", summary)


if __name__ == "__main__":
    main()