"""Reproduce the paper's MOrangeT sensitivity analysis (its Tables 3 and 4): GT boxes, each kept with probability `rate`.

    python -m reloc3d                                # Tables 3 and 4 on every downloaded sequence
    python -m reloc3d --rates 1 --no-reloc-rates     # perfect detection only
    python -m reloc3d --max-geom-error 20            # an Algorithm 1 setting the paper leaves open

Reads data/moranget/raw/<seq>/{gt/gt.txt, geom/, images/}, writes runs/<stamp>_reloc3d/ and prints ours
against the paper. Scores come from mapsort.eval (TrackEval), as for every other tracker in this repo.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mapsort import runs
from mapsort.data import Sequence, data_root

from . import Params, Reloc3D, load_cameras

PAPER = Path(__file__).with_name("paper_tables.csv")
FIELDS = ("HOTA", "DetA", "AssA", "MOTA")
FIXED = ("relocalization", "seed")  # set per configuration, not by flags


def raw_sequence(raw: Path, name: str) -> Sequence:
    path = raw / name
    return Sequence(dataset="moranget", split="test", name=name, path=path, dets_path=None,
                    landmark_dir=path, landmark_backends=(), recon_path=path / "geom")


def gt_boxes(seq: Sequence, rate: float, seed: int) -> list[np.ndarray]:
    """xyxy GT boxes per frame; each row survives when its U(0, 1) draw is <= rate (Sec. 3.3.7)."""
    rows = seq.gt_rows()
    rows = rows[np.random.default_rng(seed).random(len(rows)) <= rate]
    out = [np.empty((0, 4)) for _ in range(seq.num_frames)]
    for f in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == f]
        out[f - 1] = np.c_[r[:, 2:4], r[:, 2:4] + r[:, 4:6]]
    return out


def track(job: tuple) -> str:
    raw, name, params, rate, path = job
    seq = raw_sequence(raw, name)
    names = sorted(p.name for p in (seq.path / "images").glob("*.jpg"))
    if len(names) != seq.num_frames:
        raise SystemExit(f"{name}: {len(names)} images but seqLength {seq.num_frames}")
    tracks = Reloc3D(load_cameras(seq.recon_path, names), params).run(gt_boxes(seq, rate, params.seed))
    lines = [
        f"{f + 1},{k},{b[0]:.2f},{b[1]:.2f},{b[2] - b[0]:.2f},{b[3] - b[1]:.2f},1,1,-1"
        for k, t in enumerate(tracks, 1)
        for f, b in sorted(zip(t.frames, t.boxes), key=lambda fb: fb[0])
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" if lines else "")
    return name


def scores(result: dict) -> dict[str, float]:
    return {**{k: result[k] for k in FIELDS}, "CbyT": result["IDs"], "CbyT_GT": result["GT_IDs"]}


def paper_table(reloc: bool, rate: float, names: list[str], gt_boxes_per_seq: dict[str, int]) -> dict[str, dict]:
    """The paper's rows; its All pools 12 sequences, so over fewer only the count and MOTA are restated exactly."""
    with PAPER.open() as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))
                if int(r["relocalization"]) == reloc and float(r["rate"]) == rate]
    out = {r["sequence"]: {k: float(v) for k, v in r.items() if k in (*FIELDS, "CbyT", "CbyT_GT")} for r in rows}
    if out and set(names) != set(out) - {"All"}:
        boxes = sum(gt_boxes_per_seq[n] for n in names)
        out["All"] = {
            "CbyT": sum(out[n]["CbyT"] for n in names),
            "CbyT_GT": sum(out[n]["CbyT_GT"] for n in names),
            "MOTA": sum(out[n]["MOTA"] * gt_boxes_per_seq[n] for n in names) / boxes,
        }
    return out


def report(metrics: dict, keys: list[str], paper: dict[str, dict], names: list[str]) -> None:
    print(f"{'seq':6s}" + "".join(f"{f:>17s}" for f in FIELDS) + f"{'CbyT':>14s}{'GT':>6s}{'rel. err. %':>17s}")
    errs, perrs = [], []
    for n in [*names, "All"]:
        ours = {f: float(np.mean([metrics[k][n][f] for k in keys])) for f in (*FIELDS, "CbyT")}
        gt, p = metrics[keys[0]][n]["CbyT_GT"], paper.get(n, {})
        err = 100 * abs(ours["CbyT"] - gt) / gt
        perr = 100 * abs(p["CbyT"] - p["CbyT_GT"]) / p["CbyT_GT"] if p else float("nan")
        if n != "All":
            errs.append(err)
            perrs.append(perr)
        cells = "".join(f"{ours[f]:7.2f} [{100 * p[f]:6.2f}]" if f in p else f"{ours[f]:7.2f} [   -- ]" for f in FIELDS)
        print(f"{n:6s}{cells}{ours['CbyT']:7.1f} [{p.get('CbyT', float('nan')):4.0f}]{gt:6d}{err:8.2f} [{perr:6.2f}]")
    print(f"{'median':6s}{'':68s}{'':26s}{np.median(errs):8.2f} [{np.nanmedian(perrs):6.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=data_root() / "moranget" / "raw")
    ap.add_argument("--seqs", nargs="+", help="default: every V<nn> under --raw")
    ap.add_argument("--rates", nargs="*", type=float, default=[1.0, 0.8, 0.6, 0.4])
    ap.add_argument("--no-reloc-rates", nargs="*", type=float, default=[1.0, 0.8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="box drops; rate 1 uses the first")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix of the run directory")
    for f in dataclasses.fields(Params):
        if f.name in FIXED:
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    args = ap.parse_args()

    from mapsort.eval.tracking import evaluate

    names = args.seqs or sorted(p.name for p in args.raw.glob("V[0-9][0-9]") if (p / "gt" / "gt.txt").exists())
    base = {f.name: getattr(args, f.name) for f in dataclasses.fields(Params) if f.name not in FIXED}
    seeds = lambda rate: args.seeds if rate < 1 else args.seeds[:1]  # noqa: E731
    configs = [(True, r, s) for r in args.rates for s in seeds(r)]
    configs += [(False, r, s) for r in args.no_reloc_rates for s in seeds(r)]
    key = lambda reloc, rate, seed: f"{'reloc' if reloc else 'noreloc'}_r{rate:g}_s{seed}"  # noqa: E731
    run_dir = runs.create("reloc3d" + (f"_{args.tag}" if args.tag else ""), config={
        "params": base, "sequences": names, "raw": str(args.raw), "configs": [key(*c) for c in configs],
    })

    jobs = [
        (args.raw, n, Params(**base, relocalization=reloc, seed=seed), rate,
         run_dir / "artifacts" / key(reloc, rate, seed) / f"{n}.txt")
        for reloc, rate, seed in configs
        for n in names
    ]
    # Spawned workers with one BLAS thread each: the 6x4 SVDs gain nothing from threads, and 8 pools thrash.
    os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, _ in enumerate(pool.map(track, jobs), 1):
            print(f"\r{i}/{len(jobs)} tracked", end="", flush=True)
    print()

    seqs = [raw_sequence(args.raw, n) for n in names]
    metrics = {}
    for config in configs:
        k = key(*config)
        res = evaluate(seqs, {n: run_dir / "artifacts" / k / f"{n}.txt" for n in names}, thresholds=[0.5])
        metrics[k] = {"All": scores(res["combined"]), **{n: scores(r) for n, r in res["per_sequence"].items()}}
    runs.write_json(run_dir / "metrics.json", metrics)

    gt_boxes_per_seq = {s.name: len(s.gt_rows()) for s in seqs}
    for reloc, rates in ((True, args.rates), (False, args.no_reloc_rates)):
        for rate in rates:
            keys = [key(reloc, rate, s) for s in seeds(rate)]
            print(f"\n== {'with' if reloc else 'WITHOUT'} relocalization, detection rate {rate:.0%}: "
                  f"ours{f' (mean of {len(keys)} seeds)' if len(keys) > 1 else ''} [paper]")
            report(metrics, keys, paper_table(reloc, rate, names, gt_boxes_per_seq), names)
    print(f"\nrun dir: {run_dir}")

