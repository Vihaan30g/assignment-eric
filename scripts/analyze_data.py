import re, yaml, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
base = Path(cfg["path"])
names = cfg["names"]
splits = {"train": "train", "valid": "valid", "test": "test"}

def source_id(fname):
    # Roboflow names look like  originalname_jpg.rf.<hash>.jpg
    return re.split(r"\.rf\.", fname)[0]

stems, stats = {}, {}
for s in splits:
    imgs = list((base / s / "images").glob("*"))
    stems[s] = {source_id(p.name) for p in imgs}
    counts, sizes = collections.Counter(), []
    for lf in (base / s / "labels").glob("*.txt"):
        for line in lf.read_text().splitlines():
            c, x, y, w, h = line.split()[:5]
            counts[int(c)] += 1
            sizes.append((float(w), float(h)))
    stats[s] = (len(imgs), counts, np.array(sizes))
    print(f"\n{s}: {len(imgs)} images")
    for k in sorted(counts):
        print(f"  {names[k]:15s} {counts[k]}")

print("\nLeakage check (shared source images):")
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    print(f"  {a} vs {b}: {len(stems[a] & stems[b])} shared")

# box size plot (train)
sz = stats["train"][2]
plt.figure(figsize=(6, 4))
plt.hist(np.sqrt(sz[:, 0] * sz[:, 1]), bins=50)
plt.xlabel("sqrt(box area) as fraction of image"); plt.ylabel("count")
plt.title("Box size distribution (train)")
out = ROOT / "results/images"; out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "box_sizes.png", dpi=150, bbox_inches="tight")
print("saved results/images/box_sizes.png")