"""Evaluate the trained sign-classifier checkpoint on its test split."""
import argparse
from pathlib import Path
from ultralytics import YOLO

# Get the absolute path to the assignment-eric root folder
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description="Test trained YOLO classification weights.")
    p.add_argument("--model", type=str, default="sign_n_best.pt", help="Name of the weights file in the weights/ folder")
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "signs_cls"), help="Path to classification dataset")
    p.add_argument("--imgsz", type=int, default=224, help="Image size (must match training)")
    a = p.parse_args()

    weights_path = ROOT / "weights" / a.model
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        return

    print(f"Loading model: {weights_path.name}...")
    model = YOLO(str(weights_path))

    print(f"\nEvaluating on test split of: {a.data}")
    # Using model.val() with split='test' automatically tests the model against the test folder
    metrics = model.val(
        data=a.data, 
        split="test", 
        imgsz=a.imgsz,
        plots=True, # Will generate confusion matrices and batch visualizations
        project=str(ROOT / "runs"), 
        name=f"{a.model.split('.')[0]}_eval"
    )

    print("\n" + "="*40)
    print(f"TEST RESULTS: {a.model}")
    print("="*40)
    print(f"Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
    print(f"Top-5 Accuracy: {metrics.top5 * 100:.2f}%")
    print(f"Classes evaluated: {list(model.names.values())}")
    print(f"Visualizations saved to: {metrics.save_dir}")
    print("="*40)

if __name__ == "__main__":
    main()