"""Fine-tune OSNet-x0.25 (MSMT17) per dataset on a capped, seeded sample of GT crops.

Crops are cut and resized with boxmot's own inference functions, so the model trains on what it
will see. Epoch 1 fits only the new classifier on frozen features; later epochs train everything
with label-smoothed cross-entropy plus batch-hard triplet. The epoch with the best val mAP is kept,
and epoch 0 (the MSMT17 weights unchanged) is scored on the same val crops as the baseline.

Writes models/reid/osnet-x0-25-<dataset>.pt and runs/reid-<UTC stamp>/{manifest,metrics}.json.
Caps, filters and splits: config/reid_finetune.yaml.

Usage:
    python scripts/reid/finetune.py --datasets applemots grapemots lettucemots moranget
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from boxmot.reid.core.crops import extract_crops
from boxmot.reid.core.preprocessing import resize
from boxmot.reid.core.registry import ReIDModelRegistry

from mapsort.data import REPO_ROOT, sequence

CONFIG = REPO_ROOT / "config" / "reid_finetune.yaml"
ARCH = "osnet_x0_25"
INPUT_SHAPE = (256, 128)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def model_name(dataset: str) -> str:
    return f"osnet-x0-25-{dataset}"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usable(seq, f: dict) -> np.ndarray:
    """GT rows that make clean crops: big enough, clear of the frame edge, not mostly covered."""
    g = seq.gt_rows()
    width, height = int(seq.info["imWidth"]), int(seq.info["imHeight"])
    x, y, w, h = g[:, 2], g[:, 3], g[:, 4], g[:, 5]
    e = f["edge"]
    ok = (np.minimum(w, h) >= f["min_side"]) & (x >= e) & (y >= e) & (x + w <= width - e) & (y + h <= height - e)
    for frame in np.unique(g[:, 0]):
        sel = np.flatnonzero(g[:, 0] == frame)
        x1, y1, x2, y2 = x[sel], y[sel], x[sel] + w[sel], y[sel] + h[sel]
        iw = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
        ih = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
        ioa = iw * ih / np.maximum(w[sel] * h[sel], 1e-9)[:, None]
        np.fill_diagonal(ioa, 0)
        ok[sel] &= ioa.max(axis=1, initial=0) <= f["max_ioa"]
    return g[ok]


def sample(seqs, n_ids: int, per_id: int, f: dict, rng) -> list[list]:
    """`[seq, frame, label, track_id, x, y, w, h]` rows; labels are dense over the sample."""
    pools = []
    for seq in seqs:
        rows = usable(seq, f)
        tracks = {int(t): rows[rows[:, 1] == t] for t in np.unique(rows[:, 1])}
        pools.append((seq.name, {t: r for t, r in tracks.items() if len(r) >= f["min_crops"]}))

    # Round-robin, so a sequence short of identities passes its share to the others.
    quota, left = [0] * len(pools), n_ids
    while left and any(q < len(tracks) for q, (_, tracks) in zip(quota, pools)):
        for i, (_, tracks) in enumerate(pools):
            if left and quota[i] < len(tracks):
                quota[i] += 1
                left -= 1

    out, label = [], 0
    for (name, tracks), q in zip(pools, quota):
        for tid in sorted(rng.choice(sorted(tracks), q, replace=False).tolist()):
            r = tracks[tid][np.argsort(tracks[tid][:, 0], kind="stable")]
            pick = np.unique(np.linspace(0, len(r) - 1, min(per_id, len(r))).round().astype(int))
            out += [[name, int(row[0]), label, tid, *map(float, row[2:6])] for row in r[pick]]
            label += 1
    return out


def load_crops(dataset: str, items: list[list]) -> np.ndarray:
    """uint8 BGR crops at the model's input size, cut exactly as boxmot cuts them."""
    crops = np.empty((len(items), *INPUT_SHAPE, 3), np.uint8)
    groups, paths = defaultdict(list), {}
    for i, it in enumerate(items):
        groups[(it[0], it[1])].append(i)
    for (name, frame), idx in groups.items():
        if name not in paths:
            paths[name] = sequence(dataset, "train", name).image_paths()
        bgr = cv2.imread(str(paths[name][frame - 1]))
        boxes = np.array([[x, y, x + w, y + h] for *_, x, y, w, h in (items[i] for i in idx)], np.float32)
        for i, crop in zip(idx, extract_crops(boxes, bgr, INPUT_SHAPE)):
            crops[i] = resize(crop, INPUT_SHAPE)
    return crops


def to_input(crops: np.ndarray) -> torch.Tensor:
    """uint8 BGR NHWC -> normalised RGB NCHW, as boxmot's prepare_crop_batch."""
    x = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN) / STD


def augment(x: torch.Tensor, pad: int, gen: torch.Generator) -> torch.Tensor:
    n, _, h, w = x.shape
    flip = torch.rand(n, generator=gen) < 0.5
    x = F.pad(torch.where(flip[:, None, None, None], x.flip(3), x), (pad,) * 4, mode="reflect")
    dy, dx = torch.randint(0, 2 * pad + 1, (2, n), generator=gen)
    return torch.stack([x[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w] for i in range(n)])


def batch_hard_triplet(feats: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    d = torch.cdist(feats, feats)
    same = labels[:, None] == labels[None]
    return F.relu((d * same).amax(1) - d.masked_fill(same, float("inf")).amin(1) + margin).mean()


def pk_batches(labels: np.ndarray, p: int, k: int, n_batches: int, rng):
    by_id = defaultdict(list)
    for i, label in enumerate(labels):
        by_id[int(label)].append(i)
    ids = np.array(sorted(by_id))
    for _ in range(n_batches):
        chosen = rng.choice(ids, min(p, len(ids)), replace=False)
        yield np.concatenate([rng.choice(by_id[c], k, replace=len(by_id[c]) < k) for c in chosen])


@torch.inference_mode()
def score(model, crops: np.ndarray, labels: np.ndarray) -> dict:
    """Every val crop queries all the others: mAP and rank-1."""
    model.eval()
    feats = F.normalize(torch.cat([model(to_input(crops[i : i + 128])) for i in range(0, len(crops), 128)]), dim=1)
    sim = feats @ feats.T
    sim.fill_diagonal_(-float("inf"))
    order = sim.argsort(1, descending=True)[:, :-1]
    labels = torch.from_numpy(labels)
    hits = (labels[order] == labels[:, None]).float()
    precision = hits.cumsum(1) / torch.arange(1, hits.shape[1] + 1)
    ap = (precision * hits).sum(1) / hits.sum(1).clamp(min=1)
    return {"mAP": round(ap.mean().item(), 4), "rank1": round(hits[:, 0].mean().item(), 4)}


def finetune(dataset: str, cfg: dict) -> tuple[dict, dict]:
    spec, caps, f, t = cfg["datasets"][dataset], cfg["caps"], cfg["filters"], cfg["train"]
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_items = sample([sequence(dataset, "train", s) for s in spec["train"]], caps["ids"], caps["crops_per_id"], f, rng)
    val_items = sample([sequence(dataset, "train", spec["val"])], caps["val_ids"], caps["crops_per_id"], f, rng)
    start = time.time()
    x_train, x_val = load_crops(dataset, train_items), load_crops(dataset, val_items)
    y_train = np.array([it[2] for it in train_items])
    y_val = np.array([it[2] for it in val_items])
    crop_secs = time.time() - start
    n_ids = int(y_train.max()) + 1
    print(f"{dataset}  train {n_ids} ids / {len(x_train)} crops, val {int(y_val.max()) + 1} ids / {len(x_val)} crops, "
          f"cropped in {crop_secs:.0f}s", flush=True)

    model = ReIDModelRegistry.build_model(ARCH, None, num_classes=n_ids, loss="triplet", pretrained=False, use_gpu=False)
    base = torch.load(REPO_ROOT / cfg["base"], map_location="cpu")
    missing = model.load_state_dict({k: v for k, v in base.items() if not k.startswith("classifier.")}, strict=False)
    assert sorted(missing.missing_keys) == ["classifier.bias", "classifier.weight"] and not missing.unexpected_keys

    history = [{"epoch": 0, **score(model, x_val, y_val)}]
    print(f"{dataset}  epoch 0 (MSMT17)  {history[0]}", flush=True)
    n_batches = max(1, len(x_train) // (t["p"] * t["k"]))
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (caps["epochs"] - 1) * n_batches))
    best, train_start = None, time.time()
    for epoch in range(1, caps["epochs"] + 1):
        start, losses = time.time(), []
        for idx in pk_batches(y_train, t["p"], t["k"], n_batches, rng):
            x = augment(to_input(x_train[idx]), t["pad"], gen)
            y = torch.from_numpy(y_train[idx])
            if epoch == 1:
                # A random classifier's gradients would wreck the pretrained features, so fit it alone first.
                model.eval()
                with torch.no_grad():
                    feats = model(x)
                loss = F.cross_entropy(model.classifier(feats), y, label_smoothing=t["label_smoothing"])
            else:
                model.train()
                logits, feats = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=t["label_smoothing"]) + batch_hard_triplet(
                    feats, y, t["margin"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch > 1:
                sched.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), **score(model, x_val, y_val),
                        "secs": round(time.time() - start, 1)})
        print(f"{dataset}  epoch {epoch}  {history[-1]}", flush=True)
        if best is None or history[-1]["mAP"] > history[best]["mAP"]:
            best = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_secs = time.time() - train_start

    out = REPO_ROOT / "models" / "reid" / f"{model_name(dataset)}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model_name": ARCH, "num_classes": n_ids}, out)
    print(f"{dataset}  best epoch {best}  {history[best]}  vs MSMT17 {history[0]}  -> {out}", flush=True)

    def split_record(names, items):
        return {"sequences": names, "ids": len({it[2] for it in items}), "crops": len(items),
                "items": ["seq,frame,label,track_id,x,y,w,h", *(",".join(f"{v:g}" if isinstance(v, float) else str(v)
                                                                       for v in it) for it in items)]}

    manifest = {
        "model": model_name(dataset),
        "weights": str(out.relative_to(REPO_ROOT)),
        "weights_sha256": sha256(out),
        "base_sha256": sha256(REPO_ROOT / cfg["base"]),
        "train": split_record(spec["train"], train_items),
        "val": split_record([spec["val"]], val_items),
    }
    metrics = {"best_epoch": best, "baseline": history[0], "best": history[best], "history": history,
               "secs": {"crops": round(crop_secs, 1), "train": round(train_secs, 1)}}
    return manifest, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    run_dir = REPO_ROOT / "runs" / f"reid-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"script": "scripts/reid/finetune.py", "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")), "torch": torch.__version__, "config": cfg, "datasets": {}}
    metrics = {}
    for dataset in args.datasets:
        manifest["datasets"][dataset], metrics[dataset] = finetune(dataset, cfg)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
