"""Pool all images, group near-duplicates, split by group (stratified by class)."""
import argparse, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
FRACS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def source_id(name):
    return name.split(".rf.")[0]


def dhash(path, n=16):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()


def clean_label(path):
    """YOLO box rows; polygon rows are converted to boxes; values clamped to [0,1]."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        if len(p) > 5:
            xs = [float(v) for v in p[1::2]]
            ys = [float(v) for v in p[2::2]]
            k = min(len(xs), len(ys))
            x0, x1, y0, y1 = min(xs[:k]), max(xs[:k]), min(ys[:k]), max(ys[:k])
            p = [p[0], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        vals = [min(1.0, max(0.0, float(v))) for v in p[1:5]]
        if vals[2] <= 0 or vals[3] <= 0:
            continue
        rows.append(f"{int(float(p[0]))} " + " ".join(f"{v:.6f}" for v in vals))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/assignment1.v2i.yolov11"))
    ap.add_argument("--dst", default=str(ROOT / "data/assignment1_clean"))
    ap.add_argument("--thr", type=int, default=10, help="max hash distance (of 256) to call two images duplicates")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    rng = random.Random(a.seed)

    names = yaml.safe_load(open(src / "data.yaml"))["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))

    # 1. load pool
    items, hashes = [], []
    for s in SPLITS:
        for img in sorted((src / s / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            h = dhash(img)
            if h is None:
                print("skipping unreadable image:", img.name)
                continue
            rows = clean_label(src / s / "labels" / (img.stem + ".txt"))
            items.append({"img": img, "rows": rows,
                          "cls": [int(r.split()[0]) for r in rows]})
            hashes.append(h)
    n = len(items)
    print(f"Loaded {n} images")

    # 2. group duplicates with union-find
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

    first_by_root = {}
    for i, it in enumerate(items):          # same filename root
        r = source_id(it["img"].name)
        if r in first_by_root:
            union(first_by_root[r], i)
        else:
            first_by_root[r] = i

    A = np.array(hashes, dtype=np.float32)  # near-identical images
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
    print(f"{len(groups)} groups. Largest group sizes: {sizes[:8]}")
    print(f"Largest group = {100 * sizes[0] / n:.1f}% of all images")

    # contact sheet of the largest group, to eyeball it
    big = max(groups.values(), key=len)
    tiles = [cv2.resize(cv2.imread(str(items[i]["img"])), (320, 240)) for i in big[:8]]
    while len(tiles) % 4:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    out_img = ROOT / "results" / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img / "largest_duplicate_group.jpg"), sheet)

    # 3. stratified split by group (rarest class first)
    total = Counter(c for it in items for c in it["cls"])
    rank = {c: i for i, c in enumerate(sorted(total, key=total.get))}
    desired = {s: {c: total[c] * FRACS[s] for c in total} for s in SPLITS}
    desired_imgs = {s: n * FRACS[s] for s in SPLITS}

    ginfo = []
    for g in groups.values():
        cc = Counter(c for i in g for c in items[i]["cls"])
        key = (min(rank[c] for c in cc) if cc else 99, rng.random())
        ginfo.append((key, g, cc))
    ginfo.sort(key=lambda t: t[0])

    assign = {}
    for _, g, cc in ginfo:
        if cc:
            rare = min(cc, key=lambda c: total[c])
            best = max(SPLITS, key=lambda s: (desired[s][rare],
                                              sum(desired[s][c] for c in cc),
                                              desired_imgs[s]))
        else:
            best = max(SPLITS, key=lambda s: desired_imgs[s])
        for i in g:
            assign[i] = best
        for c, v in cc.items():
            desired[best][c] -= v
        desired_imgs[best] -= len(g)

    # 4. write dataset
    if dst.exists():
        shutil.rmtree(dst)
    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True)
        (dst / s / "labels").mkdir(parents=True)
    n_img, n_inst = Counter(), {s: Counter() for s in SPLITS}
    for i, it in enumerate(items):
        s = assign[i]
        shutil.copy2(it["img"], dst / s / "images" / it["img"].name)
        (dst / s / "labels" / (it["img"].stem + ".txt")).write_text(
            "\n".join(it["rows"]) + ("\n" if it["rows"] else ""))
        n_img[s] += 1
        n_inst[s].update(it["cls"])

    cfg = {"path": str(dst), "train": "train/images", "val": "valid/images",
           "test": "test/images", "names": {int(k): v for k, v in names.items()}}
    yaml.safe_dump(cfg, open(dst / "data.yaml", "w"), sort_keys=False)
    yaml.safe_dump(cfg, open(ROOT / "configs" / "data.yaml", "w"), sort_keys=False)

    print("\nFinal split:")
    for s in SPLITS:
        print(f"  {s}: {n_img[s]} images", {names[c]: v for c, v in sorted(n_inst[s].items())})
        for c in names:
            if n_inst[s][c] < 30:
                print(f"    WARNING: only {n_inst[s][c]} '{names[c]}' instances in {s}")
    print("\nWrote", dst, "and updated configs/data.yaml")


if __name__ == "__main__":
    main()  