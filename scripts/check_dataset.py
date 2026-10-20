import re, sys, collections
from pathlib import Path
import cv2, numpy as np, yaml

root = Path(sys.argv[1])
names = yaml.safe_load(open(root / "data.yaml"))["names"]
if isinstance(names, list):
    names = dict(enumerate(names))
SPLITS = ("train", "valid", "test")

# 1. counts per split
print("== Images and instances per split ==")
for s in SPLITS:
    imgs = list((root / s / "images").iterdir())
    cnt = collections.Counter()
    for lf in (root / s / "labels").glob("*.txt"):
        for line in lf.read_text().splitlines():
            if line.strip():
                cnt[int(line.split()[0])] += 1
    print(f"{s}: {len(imgs)} images", {names[k]: v for k, v in sorted(cnt.items())})

# 2. shared filenames
print("\n== Shared filenames across splits ==")
def src(n): return re.split(r"\.rf\.", n)[0]
stems = {s: {src(p.name) for p in (root / s / "images").iterdir()} for s in SPLITS}
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    print(f"{a} vs {b}: {len(stems[a] & stems[b])} shared")

# 3. near-duplicate images (even if renamed)
print("\n== Near-duplicate images across splits (hash check) ==")
def dhash(p, n=16):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return np.zeros(n * n, dtype=bool)
    im = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()

H, owner = [], []
for s in SPLITS:
    for p in sorted((root / s / "images").iterdir()):
        H.append(dhash(p)); owner.append((s, p.name))
H = np.array(H)

pairs = collections.Counter()
examples = []
for i, (s, n) in enumerate(owner):
    if s == "train":
        continue
    d = (H != H[i]).sum(1)
    for j in np.where(d <= 10)[0]:
        if j != i and owner[j][0] != s:
            pairs[(s, owner[j][0])] += 1
            if len(examples) < 5:
                examples.append((owner[i], owner[j], int(d[j])))

if not pairs:
    print("None found. Clean.")
else:
    for (a, b), c in pairs.items():
        print(f"{a} image matches {b} image: {c} cases")
    print("Examples to look at by eye:")
    for e in examples:
        print(" ", e)