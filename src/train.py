import argparse, json, shutil
from pathlib import Path
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)   # e.g. yolo11s.pt
    p.add_argument("--name", required=True)    # e.g. y11s
    p.add_argument("--data", default=str(ROOT / "configs/data.yaml"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    settings.update({"tensorboard": True})

    if a.resume:
        last = ROOT / "runs" / a.name / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        local = ROOT / "pretrained" / a.model
        model = YOLO(str(local) if local.exists() else a.model)
        model.train(
            data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            patience=a.patience, seed=a.seed, plots=True, save_period=10,
            project=str(ROOT / "runs"), name=a.name,
        )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    # final evaluation on the held-out TEST split
    m = YOLO(str(best)).val(data=a.data, split="test", imgsz=a.imgsz,
                            project=str(ROOT / "runs"), name=a.name + "_test")
    summary = {
        "model": a.model, "name": a.name,
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "per_class_mAP50-95": {m.names[i]: float(v) for i, v in enumerate(m.box.maps)},
    }

    out = ROOT / "results" / "training" / a.name
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.png", "*.yaml"):
        for f in save_dir.glob(pattern):
            shutil.copy(f, out / f.name)
    (out / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "weights" / f"{a.name}_best.pt")
    print(f"Done. Test mAP50-95 = {summary['mAP50-95']:.3f}")

if __name__ == "__main__":
    main()