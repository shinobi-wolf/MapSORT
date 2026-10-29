#!/usr/bin/env python3
"""Paper figures: sample frames, tracker comparisons, identity timelines, the per-frame camera, failure cases.

    python scripts/viz/paper_figures.py capture --dataset moranget --seq V01   # MapSORT state per frame
    python scripts/viz/paper_figures.py samples|compare|timelines|camera|failures

Tracker outputs are read from the paper's eval runs; `capture` re-runs MapSORT exactly as eval.py does
and checks that its output matches the stored run before saving the per-frame camera and track states.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from mapsort.data import sequence  # noqa: E402

FIG = REPO / "docs" / "precision-ag-paper" / "figures"
CACHE = REPO / "runs" / "paper-figures" / "artifacts"
RUNS = {
    "BoT-SORT": REPO / "runs" / "eval-20261024-000943" / "artifacts" / "botsort-a1234f7c",
    "MapSORT": REPO / "runs" / "eval-20261024-000943" / "artifacts" / "mapsort-e4c91726",
}
REPRESENTATIVE = {"lettucemots": "0011", "moranget": "V01", "grapemots": "PathPlanning_7", "applemots": "0008"}
NAMES = {"lettucemots": "LettuceMOTS", "moranget": "MOrangeT", "grapemots": "GrapeMOTS", "applemots": "AppleMOTS"}
SAMPLE_FRAMES = {"lettucemots": 60, "moranget": 200, "grapemots": 100, "applemots": 150}
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COLOURS = {"BoT-SORT": (200, 40, 40), "MapSORT": (31, 92, 153)}
OK, BAD, MISS = (31, 92, 153), (214, 69, 65), (205, 205, 205)


# ── data ─────────────────────────────────────────────────────────────────────


def read_mot(path: Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    """frame -> [(id, xyxy)]."""
    out: dict[int, list] = defaultdict(list)
    for row in np.atleast_2d(np.loadtxt(path, delimiter=",")):
        x, y, w, h = row[2:6]
        out[int(row[0])].append((int(row[1]), np.array([x, y, x + w, y + h])))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo, hi = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(hi - lo, 0, None).prod(2)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(1)  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def assign(gt: dict, pred: dict, num_frames: int, threshold: float = 0.5) -> list[dict[int, int]]:
    """Per-frame {gt id: tracker id}, matched as CLEAR does (mapsort/eval/revisit.py)."""
    prev: dict[int, int] = {}
    out = []
    for f in range(1, num_frames + 1):
        g, p = gt.get(f, []), pred.get(f, [])
        cur: dict[int, int] = {}
        if g and p:
            sim = iou(np.array([b for _, b in g]), np.array([b for _, b in p]))
            keep_prev = np.array([[prev.get(gi) == pi for pi, _ in p] for gi, _ in g])
            score = 1000 * keep_prev + sim
            score[sim < threshold - 1e-12] = 0
            rows, cols = linear_sum_assignment(-score)
            cur = {g[r][0]: p[c][0] for r, c in zip(rows, cols) if score[r, c] > 1e-12}
        prev = cur
        out.append(cur)
    return out


def load(ds: str, name: str, tracker: str):
    seq = sequence(ds, "test", name)
    return seq, read_mot(seq.gt_path), read_mot(RUNS[tracker] / ds / f"{name}.txt")


def first_appearance(pred: dict) -> dict[int, int]:
    """Tracker id -> 1-based rank of its first appearance, as the identities are numbered in the figures."""
    order: dict[int, int] = {}
    for f in sorted(pred):
        for tid, _ in sorted(pred[f], key=lambda r: (r[1][1], r[1][0])):
            order.setdefault(tid, len(order) + 1)
    return order


def frame_image(seq, f: int) -> Image.Image:
    return Image.open(seq.image_paths()[f - 1]).convert("RGB")


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def tag(draw: ImageDraw.ImageDraw, xy, text: str, colour, size: int) -> None:
    f = font(size)
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    x, y = xy
    pad = max(2, size // 6)
    y = max(y - (b - t) - 2 * pad, 0)
    draw.rectangle([x, y, x + (r - l) + 2 * pad, y + (b - t) + 2 * pad], fill=colour)
    draw.text((x + pad - l, y + pad - t), text, font=f, fill=(255, 255, 255))


def header(image: Image.Image, title: str, subtitle: str | list[str], colour, scale: float) -> Image.Image:
    small = font(int(24 * scale), bold=False)
    lines = []
    for text in [subtitle] if isinstance(subtitle, str) else subtitle:
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and small.getlength(trial) > image.width - 28 * scale:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    h = int((60 + 32 * len(lines) + 12) * scale)
    out = Image.new("RGB", (image.width, image.height + h), (255, 255, 255))
    out.paste(image, (0, h))
    d = ImageDraw.Draw(out)
    d.text((int(14 * scale), int(8 * scale)), title, font=font(int(34 * scale)), fill=(25, 25, 25))
    for k, line in enumerate(lines):
        d.text((int(14 * scale), int((56 + 32 * k) * scale)), line, font=font(int(24 * scale), bold=False), fill=colour)
    return out


def hstack(images: list[Image.Image], gap: int, fill=(255, 255, 255)) -> Image.Image:
    h = max(i.height for i in images)
    out = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), h), fill)
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vstack(images: list[Image.Image], gap: int, fill=(255, 255, 255)) -> Image.Image:
    w = max(i.width for i in images)
    out = Image.new("RGB", (w, sum(i.height for i in images) + gap * (len(images) - 1)), fill)
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def fit(image: Image.Image, long_side: int) -> Image.Image:
    s = long_side / max(image.size)
    return image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)


# ── Fig. samples ─────────────────────────────────────────────────────────────


def samples(_args) -> None:
    for ds, name in REPRESENTATIVE.items():
        seq, gt, _ = load(ds, name, "MapSORT")
        f = SAMPLE_FRAMES[ds]
        image = frame_image(seq, f)
        s = 1200 / max(image.size)
        image = fit(image, 1200)
        d = ImageDraw.Draw(image)
        width = 6
        for _, b in gt[f]:
            # The stroke lies outside the box so it does not cover small fruit.
            x1, y1, x2, y2 = b * s
            d.rectangle([x1 - width, y1 - width, x2 + width - 1, y2 + width - 1], outline=(0, 0, 0), width=width)
        out = FIG / f"sample_{ {'lettucemots': 'lettuce', 'moranget': 'orange', 'grapemots': 'grape', 'applemots': 'apple'}[ds]}.jpg"
        image.save(out, quality=92)
        print(out, image.size, len(gt[f]), "boxes")


# ── MapSORT state capture ────────────────────────────────────────────────────


def capture(args) -> None:
    import eval as ev  # scripts/eval/eval.py
    from boxmot.trackers.botsort.track import TrackState

    from mapsort import tracker as mt
    from mapsort.eval.mot_io import write_tracks

    seq = sequence(args.dataset, "test", args.seq)
    spec = json.loads((RUNS["MapSORT"] / "spec.json").read_text())
    resect = mt.MapSort._resect

    def recording_resect(self, pairs):
        P = resect(self, pairs)
        if P is not None:
            self.fig_camera = P
        return P

    mt.MapSort._resect = recording_resect
    landmarks = ev.merge({"enabled": True}, spec["hp"].get("landmarks", {}))
    stored = seq.stored_landmarks(spec["source"])
    reid = ev.encoder_for(seq, False)
    tracker = mt.Tracker(botsort=spec["hp"].get("botsort"), reid=reid, landmarks=landmarks,
                         frame_rate=int(seq.info["frameRate"]))
    dets = seq.stored_detections()
    states = []
    for i, frame in enumerate(ev.frames(seq)):
        reid.frame = i
        m = tracker.tracker
        m.fig_camera = None
        tracker(i, frame, dets[i], stored[i])
        rows = []
        seen = set()
        for kind, pool in (("active", m.active_tracks), ("lost", m.lost_stracks), ("dormant", m.dormant)):
            for t in pool:
                if t.id in seen or not t.landmark.observed:
                    continue
                seen.add(t.id)
                state = {TrackState.Tracked: "tracked", TrackState.Lost: "lost"}.get(t.state, "dormant")
                rows.append({
                    "id": int(t.id), "state": state, "confirmed": bool(t.is_activated), "matched": t.frame_id == m.frame_count,
                    "landmark": np.asarray(t.landmark.gaussian[0], float), "offset": np.asarray(t.offset, float).copy(),
                    "offsets": int(t.offsets), "wh": np.asarray(t.wh, float).copy(),
                })
        states.append({"camera": m.fig_camera, "tracks": rows, "shape": frame.shape[:2]})
        if (i + 1) % 50 == 0:
            print(f"{args.dataset}/{args.seq} {i + 1}/{seq.num_frames}", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    out = write_tracks(tracker.tracks, CACHE / f"{args.dataset}_{args.seq}.txt")
    same = out.read_text() == (RUNS["MapSORT"] / args.dataset / f"{args.seq}.txt").read_text()
    print(f"{args.dataset}/{args.seq}: output {'matches' if same else 'DIFFERS FROM'} the paper run")
    if not same:
        raise SystemExit(1)
    with open(CACHE / f"{args.dataset}_{args.seq}.pkl", "wb") as fh:
        pickle.dump(states, fh)


def load_states(ds: str, name: str) -> list[dict]:
    with open(CACHE / f"{ds}_{name}.pkl", "rb") as fh:
        return pickle.load(fh)


# ── identity bookkeeping ─────────────────────────────────────────────────────


def identity_status(gt: dict, A: list[dict], num_frames: int):
    """[objects, frames] array: 0 absent, 1 first identity, 2 another identity, 3 unmatched; rows by first appearance."""
    first_frame: dict[int, int] = {}
    for f in sorted(gt):
        for g, _ in gt[f]:
            first_frame.setdefault(g, f)
    order = sorted(first_frame, key=lambda g: (first_frame[g], g))
    row = {g: i for i, g in enumerate(order)}
    status = np.zeros((len(order), num_frames), np.uint8)
    first_id: dict[int, int] = {}
    for f in range(1, num_frames + 1):
        for g, _ in gt.get(f, []):
            t = A[f - 1].get(g)
            if t is None:
                status[row[g], f - 1] = 3
                continue
            first_id.setdefault(g, t)
            status[row[g], f - 1] = 1 if first_id[g] == t else 2
    return status


def returning(gt: dict, A: list[dict], f: int, min_gap: int = 10) -> tuple[int, int]:
    """(returned objects in view at frame f, of which still carry their first identity)."""
    last, back, first_id = {}, set(), {}
    for k in range(1, f + 1):
        for g, _ in gt.get(k, []):
            if g in last and k - last[g] >= min_gap:
                back.add(g)
            last[g] = k
        for g, t in A[k - 1].items():
            first_id.setdefault(g, t)
    here = [g for g, _ in gt.get(f, []) if g in back]
    return len(here), sum(A[f - 1].get(g) == first_id.get(g) for g in here)


# ── Fig. qual-other ──────────────────────────────────────────────────────────

COMPARE = [  # dataset, sequence, frame, object noun, crop to the objects, panel width, text scale
    ("moranget", "V01", 232, "oranges", False, 820, 1.55),
    ("grapemots", "PathPlanning_7", 221, "bunches", True, 1400, 1.3),
]


def returned_ids(gt: dict, f: int, min_gap: int = 10) -> set[int]:
    last, back = {}, set()
    for k in range(1, f + 1):
        for g, _ in gt.get(k, []):
            if g in last and k - last[g] >= min_gap:
                back.add(g)
            last[g] = k
    return {g for g, _ in gt.get(f, []) if g in back}


def tracker_panel(seq, gt, pred, f: int, tracker: str, noun: str, crop, scale: float, text: float) -> Image.Image:
    """Every box the tracker reports at frame f; the returning objects are tagged with their identity."""
    A = assign(gt, pred, seq.num_frames)
    order = first_appearance(pred)
    issued = len({t for k in range(1, f + 1) for t, _ in pred.get(k, [])})
    back, kept = returning(gt, A, f)
    shown = {A[f - 1][g] for g in returned_ids(gt, f) if g in A[f - 1]}
    image = frame_image(seq, f)
    if crop is not None:
        image = image.crop(crop)
    off = np.array(crop[:2] if crop is not None else (0, 0), float)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(image)
    colour = COLOURS[tracker]
    rows = sorted(pred.get(f, []), key=lambda r: -r[1][1])
    for t, b in rows:
        b = (b - np.r_[off, off]) * scale
        d.rectangle(b.tolist(), outline=colour, width=6 if t in shown else 2)
    for t, b in rows:
        if t in shown:
            b = (b - np.r_[off, off]) * scale
            tag(d, (b[0], b[1]), str(order[t]), colour, int(24 * text))
    lines = [f"{issued} identities issued so far", f"{kept} of {back} returning {noun} keep their first identity"]
    return header(image, tracker, lines, colour, text)


def compare(_args) -> None:
    for ds, name, f, noun, crop_to, width_px, text in COMPARE:
        seq, gt, _ = load(ds, name, "MapSORT")
        w, h = Image.open(seq.image_paths()[0]).size
        crop = None
        if crop_to:
            boxes = np.array([b for _, b in gt[f]])
            lo, hi = boxes[:, :2].min(0) - [0.03 * w, 0.06 * h], boxes[:, 2:].max(0) + [0.03 * w, 0.03 * h]
            crop = tuple(int(v) for v in (*np.maximum(lo, 0), *np.minimum(hi, [w, h])))
        panels = []
        for tracker in ("BoT-SORT", "MapSORT"):
            _, _, pred = load(ds, name, tracker)
            width = (crop[2] - crop[0]) if crop else w
            panels.append(tracker_panel(seq, gt, pred, f, tracker, noun, crop, width_px / width, text))
        out = (vstack if crop_to else hstack)(panels, 18)
        path = FIG / f"qual_{NAMES[ds].lower()}.jpg"
        out.save(path, quality=92)
        print(path, out.size)


# ── Fig. timelines ───────────────────────────────────────────────────────────


def timelines(_args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    cmap = ListedColormap(["white", np.array(OK) / 255, np.array(BAD) / 255, np.array(MISS) / 255])
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.3), gridspec_kw={"hspace": 0.32, "wspace": 0.28})
    for c, (ds, name) in enumerate(REPRESENTATIVE.items()):
        seq, gt, _ = load(ds, name, "MapSORT")
        for r, tracker in enumerate(("BoT-SORT", "MapSORT")):
            _, _, pred = load(ds, name, tracker)
            status = identity_status(gt, assign(gt, pred, seq.num_frames), seq.num_frames)
            ax = axes[r, c]
            ax.imshow(status, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
            issued = len({t for rows in pred.values() for t, _ in rows})
            ax.set_title(f"{tracker}: {issued} IDs for {status.shape[0]}", fontsize=6.5, pad=2.5)
            ax.tick_params(length=2, pad=1.5, labelsize=6)
            if r == 0:
                ax.text(0.5, 1.2, f"{NAMES[ds]} {name.replace('PathPlanning_', 'PP_')}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")
            if r == 1:
                ax.set_xlabel("Frame", labelpad=1)
            if c == 0:
                ax.set_ylabel("Object", labelpad=1)
    fig.legend(handles=[Patch(color=np.array(OK) / 255, label="first identity"),
                        Patch(color=np.array(BAD) / 255, label="a later identity"),
                        Patch(color=np.array(MISS) / 255, label="not tracked")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035), fontsize=7)
    path = FIG / "timelines.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print(path)


# ── Fig. camera ──────────────────────────────────────────────────────────────


def project(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.c_[X, np.ones(len(X))] @ P.T
    return h[:, :2] / h[:, 2:3], h[:, 2]


def predicted(P: np.ndarray, tracks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Predicted xyxy boxes (landmark projection plus learned offset, last full size) and projective depths."""
    if not tracks:
        return np.empty((0, 4)), np.empty(0)
    centre, depth = project(P, np.array([t["landmark"] for t in tracks]))
    wh = np.array([t["wh"] for t in tracks])
    centre = centre + np.array([t["offset"] for t in tracks]) * np.sqrt(wh.prod(1))[:, None]
    return np.c_[centre - wh / 2, centre + wh / 2], depth


def dashed(d: ImageDraw.ImageDraw, box, colour, width: int, dash: int) -> None:
    x1, y1, x2, y2 = box
    for (a, b), (c, e) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(int(np.hypot(c - a, e - b) // (2 * dash)), 1)
        for k in range(n):
            s, t = k / n, (k + 0.5) / n
            d.line([(a + (c - a) * s, b + (e - b) * s), (a + (c - a) * t, b + (e - b) * t)], fill=colour, width=width)


CAMERA_FRAMES = {"lettucemots": None, "moranget": None, "grapemots": None, "applemots": 150}
# Image-coordinate window shown, where the fruit are too small to read in the whole frame.
CAMERA_VIEWS = {"moranget": np.array([430, 480, 1730, 2130]), "applemots": np.array([-30, -40, 2560, 1010])}
REVIVE, WAIT = (46, 125, 50), (196, 110, 20)


def camera_frame(states: list[dict], horizon: int = 8) -> int:
    """A frame whose camera places many lost and dormant tracks near the view, several revived soon after."""
    best, best_score = 0, -1.0
    for i, s in enumerate(states):
        if s["camera"] is None or i + horizon >= len(states):
            continue
        cands = [t for t in s["tracks"] if t["state"] != "tracked" and t["offsets"] >= 2]
        if not cands:
            continue
        h, w = s["shape"]
        boxes, _ = predicted(s["camera"], cands)
        c = (boxes[:, :2] + boxes[:, 2:]) / 2
        near = ((c > -0.3 * np.array([w, h])) & (c < 1.3 * np.array([w, h]))).all(1)
        later = {t["id"] for k in range(i + 1, i + 1 + horizon) for t in states[k]["tracks"] if t["matched"]}
        soon = sum(t["id"] in later for t, ok in zip(cands, near) if ok)
        score = 3 * soon + near.sum() + 0.2 * len(s["tracks"])
        if score > best_score:
            best, best_score = i, score
    return best + 1


def camera_panel(ds: str, name: str, f: int | None, horizon: int = 8) -> tuple[Image.Image, int, dict]:
    seq, _, pred = load(ds, name, "MapSORT")
    states = load_states(ds, name)
    f = f or camera_frame(states, horizon)
    s = states[f - 1]
    P, (h, w) = s["camera"], s["shape"]
    reported = dict(pred.get(f, []))
    tracked = [t for t in s["tracks"] if t["matched"] and t["id"] in reported]
    cands = [t for t in s["tracks"] if t["state"] != "tracked" and t["offsets"] >= 2]
    later = {t["id"] for k in range(f, min(f + horizon, len(states))) for t in states[k]["tracks"] if t["matched"]}
    _, depth_tracked = predicted(P, tracked)
    sign = np.sign(np.median(depth_tracked)) if len(depth_tracked) else 1.0
    boxes, depth = predicted(P, cands)

    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    view = CAMERA_VIEWS.get(ds)
    if view is None:
        cap = 0.35 * np.array([w, h])
        keep = (centre > -cap) & (centre < np.array([w, h]) + cap)
    else:
        keep = (centre > view[:2]) & (centre < view[2:])
    keep = (np.sign(depth) == sign) & keep.all(1)
    cands, boxes = [t for t, k in zip(cands, keep) if k], boxes[keep]
    if view is None:
        pad = 0.03 * max(w, h)
        lo = np.minimum([0, 0], boxes[:, :2].min(0) if len(boxes) else [0, 0]) - pad
        hi = np.maximum([w, h], boxes[:, 2:].max(0) if len(boxes) else [w, h]) + pad
    else:
        lo, hi = np.array(view[:2], float), np.array(view[2:], float)
    scale = 1500 / max(hi - lo)
    W, H = (np.ceil((hi - lo) * scale)).astype(int)
    canvas = Image.new("RGB", (int(W), int(H)), (236, 236, 236))
    image = frame_image(seq, f).resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    canvas.paste(image, tuple(np.round(-lo * scale).astype(int)))
    d = ImageDraw.Draw(canvas)
    to_canvas = lambda b: ((np.asarray(b, float).reshape(-1, 2) - lo) * scale).ravel().tolist()  # noqa: E731
    d.rectangle(to_canvas([0, 0, w - 1, h - 1]), outline=(60, 60, 60), width=2)
    for t in tracked:
        d.rectangle(to_canvas(reported[t["id"]]), outline=(0, 0, 0), width=4)
    counts = {"inside": 0, "outside": 0, "revived": 0}
    for t, b in sorted(zip(cands, boxes), key=lambda r: r[0]["id"] in later):
        c = (b[:2] + b[2:]) / 2
        counts["inside" if (0 <= c[0] < w and 0 <= c[1] < h) else "outside"] += 1
        soon = t["id"] in later
        counts["revived"] += soon
        x1, y1, x2, y2 = to_canvas(b)
        grow = max(0.0, 14 - min(x2 - x1, y2 - y1)) / 2  # keep tiny fruit legible
        d.rectangle([x1 - grow, y1 - grow, x2 + grow, y2 + grow], outline=REVIVE if soon else WAIT, width=5)
    return canvas, f, counts


def camera(_args) -> None:
    for ds, name in REPRESENTATIVE.items():
        panel, f, counts = camera_panel(ds, name, CAMERA_FRAMES[ds])
        path = FIG / f"camera_{NAMES[ds].lower()}.jpg"
        panel.save(path, quality=92)
        print(path, panel.size, "frame", f, counts)


# ── Fig. failures ────────────────────────────────────────────────────────────

GRAPE_CASE = ("grapemots", "PathPlanning_3", (86, 185), ({14, 15, 16, 17}, {20, 21, 22, 23}))
APPLE_CASE = ("applemots", "0007", 91)  # a lost MapSORT track matched, where the camera places it, to another apple
LEGEND = "black: ground-truth identity; blue: MapSORT identity"


def window(centre: np.ndarray, half: np.ndarray, w: int, h: int) -> np.ndarray:
    """Top-left corner of a 2*half crop around centre, kept inside the frame."""
    return np.clip(centre - half, 0, np.array([w, h]) - 2 * half)


def overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def paired_tags(d: ImageDraw.ImageDraw, items: list, size: int, bounds) -> None:
    """Per box, a black ground-truth tag and a blue MapSORT tag side by side, placed clear of the other boxes."""
    f = font(size)
    pad = max(2, size // 6)
    occupied = [list(b) for b, _ in items]
    for b, parts in items:
        widths = [d.textbbox((0, 0), t, font=f)[2] + 2 * pad for t, _ in parts]
        tw, th = sum(widths), size + 2 * pad
        x1, y1, x2, y2 = b
        options = [(x1, y1 - th - 2), (x1, y2 + 2), (x2 + 2, y1), (x1 - tw - 2, y1), (x2 - tw, y1 - th - 2), (x2 - tw, y2 + 2)]

        def cost(o):
            r = [o[0], o[1], o[0] + tw, o[1] + th]
            outside = r[0] < bounds[0] or r[1] < bounds[1] or r[2] > bounds[2] or r[3] > bounds[3]
            return sum(overlap(r, q) for q in occupied) + (1e9 if outside else 0)

        x, y = min(options, key=cost)
        for (text, colour), width in zip(parts, widths):
            l, t, _, _ = d.textbbox((0, 0), text, font=f)
            d.rectangle([x, y, x + width, y + th], fill=colour)
            d.text((x + pad - l, y + pad - t), text, font=f, fill=(255, 255, 255))
            x += width
        occupied.append([x - tw, y, x, y + th])


def failure_grape() -> Image.Image:
    ds, name, frames, groups = GRAPE_CASE
    seq, gt, pred = load(ds, name, "MapSORT")
    A = assign(gt, pred, seq.num_frames)
    order = first_appearance(pred)
    w, h = Image.open(seq.image_paths()[0]).size
    panels = []
    for f, ids in zip(frames, groups):
        boxes = np.array([b for g, b in gt[f] if g in ids])
        half = np.array([200, 130])
        lo = window((boxes[:, :2].min(0) + boxes[:, 2:].max(0)) / 2, half, w, h)
        image = frame_image(seq, f).crop((*lo.astype(int), *(lo + 2 * half).astype(int)))
        s = 2.4
        image = image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
        d = ImageDraw.Draw(image)
        items = []
        for g, b in gt[f]:
            if g not in ids:
                continue
            b = (b - np.r_[lo, lo]) * s
            d.rectangle([b[0] - 4, b[1] - 4, b[2] + 3, b[3] + 3], outline=(0, 0, 0), width=4)
            parts = [(f"GT {g}", (0, 0, 0))]
            if A[f - 1].get(g) is not None:
                parts.append((str(order[A[f - 1][g]]), COLOURS["MapSORT"]))
            items.append(([b[0] - 4, b[1] - 4, b[2] + 3, b[3] + 3], parts))
        paired_tags(d, items, 22, (0, 0, image.width, image.height))
        sub = "before a motion-blurred pan" if f == frames[0] else "after it: new ground-truth identities"
        panels.append(header(image, f"Frame {f}", [sub, LEGEND], (90, 90, 90), 0.85))
    return hstack(panels, 16)


def failure_apple() -> tuple[Image.Image, dict]:
    ds, name, track = APPLE_CASE
    seq, gt, pred = load(ds, name, "MapSORT")
    A = assign(gt, pred, seq.num_frames)
    order = first_appearance(pred)
    states = load_states(ds, name)
    w, h = Image.open(seq.image_paths()[0]).size
    owned = [(f, g) for f, a in enumerate(A, 1) for g, t in a.items() if t == track]
    (fa, ga), = [owned[k] for k in range(len(owned) - 1) if owned[k + 1][1] != owned[k][1]][:1]
    fb, gb = next((f, g) for f, g in owned if g != ga)
    before = next(t for t in states[fb - 2]["tracks"] if t["id"] == track)
    box, _ = predicted(states[fb - 1]["camera"], [before])
    box = box[0]
    b_true = next(b for g, b in gt[fb] if g == gb)
    size = np.sqrt(np.prod(before["wh"]))
    others = np.array([(b[:2] + b[2:]) / 2 for g, b in gt[fb] if g != gb])
    stats = {
        "frames": (fa, fb), "gt": (ga, gb), "track": order[track], "state": before["state"],
        "error_box_sizes": float(np.linalg.norm((box[:2] + box[2:]) / 2 - (b_true[:2] + b_true[2:]) / 2) / size),
        "nearest_neighbour_box_sizes": float(np.min(np.linalg.norm(others - (b_true[:2] + b_true[2:]) / 2, axis=1)) / size),
    }
    panels = []
    for f, g in ((fa, ga), (fb, gb)):
        b0 = next(b for gg, b in gt[f] if gg == g)
        half = np.array([100, 100])
        lo = window((b0[:2] + b0[2:]) / 2, half, w, h)
        image = frame_image(seq, f).crop((*lo.astype(int), *(lo + 2 * half).astype(int)))
        s = 3.6
        image = image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
        d = ImageDraw.Draw(image)
        for gg, b in gt[f]:
            if gg != g:
                d.rectangle(((b - np.r_[lo, lo]) * s).tolist(), outline=(170, 170, 170), width=2)
        if f == fb:
            dashed(d, ((box - np.r_[lo, lo]) * s).tolist(), COLOURS["MapSORT"], 5, 9)
        b = (b0 - np.r_[lo, lo]) * s
        d.rectangle(b.tolist(), outline=(0, 0, 0), width=5)
        paired_tags(d, [(b.tolist(), [(f"GT {g}", (0, 0, 0)), (str(order[track]), COLOURS["MapSORT"])])], 24,
                    (0, 0, image.width, image.height))
        sub = f"track {order[track]} on its apple" if f == fa else \
            f"{fb - fa} frames later, on another apple at its predicted box (dashed)"
        panels.append(header(image, f"Frame {f}", [sub, LEGEND], (90, 90, 90), 0.85))
    return hstack(panels, 16), stats


def failures(_args) -> None:
    grape = failure_grape()
    path = FIG / "failure_grapemots.jpg"
    grape.save(path, quality=92)
    print(path, grape.size)
    apple, stats = failure_apple()
    path = FIG / "failure_applemots.jpg"
    apple.save(path, quality=92)
    print(path, apple.size, stats)


def main() -> None:
    figures = {"samples": samples, "capture": capture, "compare": compare, "timelines": timelines,
               "camera": camera, "failures": failures}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=list(figures))
    ap.add_argument("--dataset")
    ap.add_argument("--seq")
    args = ap.parse_args()
    figures[args.figure](args)


if __name__ == "__main__":
    main()
