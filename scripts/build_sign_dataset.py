"""Rebuild the sign-classification dataset without leakage.
  python scripts/build_sign_dataset.py [--src data/TrafficSigns_Balanced] [--thr 30]
"""
import argparse, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_MAP = {"stop": "stop", "barrierahead": "barrier_ahead", "notstop": "other"}
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SPLITS = ("train", "val", "test")
FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def root_id(name):
    return name.split(".rf.")[0]


def dhash(path, n=32):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/TrafficSigns_Balanced"))
    ap.add_argument("--dst", default=str(ROOT / "data/signs_cls"))
    ap.add_argument("--thr", type=int, default=15, help="max differing hash bits (of 1024) = duplicate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    rng = random.Random(a.seed)

    items, hashes = [], []
    for sdir in sorted(p for p in src.iterdir() if p.is_dir()):
        orig = SPLIT_MAP.get(sdir.name.lower())
        if orig is None:
            continue
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            label = CLASS_MAP.get(norm(cdir.name))
            if label is None:
                raise SystemExit(f"Unknown class folder: {cdir.name}")
            for f in sorted(cdir.rglob("*")):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                h = dhash(f)
                if h is None:
                    print("skipping unreadable:", f)
                    continue
                items.append({"path": f, "label": label, "orig": orig})
                hashes.append(h)
    n = len(items)
    print(f"Loaded {n} images")
    heights = [cv2.imread(str(it["path"])).shape[0] for it in rng.sample(items, min(300, n))]
    print(f"Median crop height (sample): {int(np.median(heights))} px")

    # --- original composition
    print("\nOriginal splits:")
    for s in SPLITS:
        c = Counter(it["label"] for it in items if it["orig"] == s)
        print(f"  {s}: {sum(c.values())} images", dict(c))

    # --- group duplicates (same source filename OR near-identical image)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    first = {}
    for i, it in enumerate(items):
        r = root_id(it["path"].name)
        if r in first:
            union(first[r], i)
        else:
            first[r] = i
    A = np.array(hashes, dtype=np.float32)
    B = 1.0 - A
    bits = A.shape[1]
    for i0 in range(0, n, 1000):
        i1 = min(n, i0 + 1000)
        dist = bits - (A[i0:i1] @ A.T + B[i0:i1] @ B.T)
        ii, jj = np.where(dist <= a.thr)
        for x, y in zip(ii + i0, jj):
            if x < y:
                union(int(x), int(y))

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    leaked = [g for g in groups.values() if len({items[i]["orig"] for i in g}) > 1]
    n_leaked = sum(len(g) for g in leaked)
    print(f"\n{len(groups)} groups; largest sizes {sizes[:6]} "
          f"(largest = {100 * sizes[0] / n:.1f}% of all images)")
    print(f"LEAKAGE: {len(leaked)} groups span several original splits "
          f"({n_leaked} images = {100 * n_leaked / n:.1f}%)")
    if sizes[0] / n > 0.05:
        print("WARNING: one group is huge; --thr may be too loose. Try --thr 15.")

    # --- stratified split by group
    total = Counter(it["label"] for it in items)
    desired = {s: {c: total[c] * FRACS[s] for c in total} for s in SPLITS}
    order = list(groups.values())
    rng.shuffle(order)
    order.sort(key=len, reverse=True)
    assign = {}
    for g in order:
        cc = Counter(items[i]["label"] for i in g)
        best = max(SPLITS, key=lambda s: sum(desired[s][c] for c in cc))
        for i in g:
            assign[i] = best
        for c, v in cc.items():
            desired[best][c] -= v

    if dst.exists():
        shutil.rmtree(dst)
    counts = {s: Counter() for s in SPLITS}
    for i, it in enumerate(items):
        s = assign[i]
        out = dst / s / it["label"]
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["path"], out / f"{i:06d}_{it['path'].name}")
        counts[s][it["label"]] += 1

    print("\nNew splits:")
    for s in SPLITS:
        print(f"  {s}: {sum(counts[s].values())} images", dict(counts[s]))
        for c in total:
            if counts[s][c] < 30:
                print(f"    WARNING: only {counts[s][c]} '{c}' images in {s}")
    print("Wrote", dst)


if __name__ == "__main__":
    main()