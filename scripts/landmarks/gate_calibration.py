"""Estimate a sequence's landmark gate from its SfM map alone, without labels.

Two detections far apart in time that are built from the same 3D points are the
same object, linked by COLMAP's own correspondences. Each side's landmark is then
re-estimated from only the points the other does not share, so the two estimates
are independent, as for a revisit that shares no points. If the landmark
covariances were honest, their Mahalanobis distance d^2 would follow chi^2_3. The
inflation k = median(d^2) / median(chi^2_3) recalibrates them, and the gate is
k * chi2inv(quantile, 3).

Writes `landmarks/<seq>/gate.json` next to `sfm.npz`.

Usage:
    python scripts/landmarks/gate_calibration.py --dataset moranget [--split test] [--seqs V01 V02]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2

from mapsort.data import data_root, splits
from sfm_mapping import read_points3D

MAX_PAIRS = 200_000


def linked_pairs(lm, min_gap: int, min_shared: int, min_frac: float, min_own: int) -> np.ndarray:
    """Mutually best point-sharing detection pairs at least min_gap frames apart, as [P, 2] rows."""
    has, frame, n_kept = lm["has_landmark"], lm["frame"], lm["n_kept"]
    ids, offsets = lm["point_ids"], lm["point_offsets"]
    rows = np.repeat(np.arange(len(frame)), np.diff(offsets))
    _, cols = np.unique(ids, return_inverse=True)
    incidence = sparse.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)))
    shared = sparse.triu(incidence @ incidence.T, k=1).tocoo()

    df = pd.DataFrame({"i": shared.row, "j": shared.col, "s": shared.data})
    df = df[has[df.i] & has[df.j]]
    df["fi"], df["fj"] = frame[df.i], frame[df.j]
    df = df[(df.fj - df.fi).abs() >= min_gap]
    df = df[(df.s >= min_shared) & (df.s >= min_frac * np.minimum(n_kept[df.i], n_kept[df.j]))]
    # Mutual best: j is i's strongest partner in j's frame, and i is j's in i's frame.
    df = df[df.s == df.groupby(["i", "fj"]).s.transform("max")]
    df = df[df.s == df.groupby(["j", "fi"]).s.transform("max")]
    df = df[(n_kept[df.i] - df.s >= min_own) & (n_kept[df.j] - df.s >= min_own)]
    return df[["i", "j"]].to_numpy()


def mahalanobis2(lm, pairs: np.ndarray, points3D: dict[int, np.ndarray]) -> np.ndarray:
    """d^2 between each pair's landmarks rebuilt from their unshared points, as sfm_mapping builds them."""
    ids, offsets, extent = lm["point_ids"], lm["point_offsets"], lm["extent"].astype(float)
    d2 = np.empty(len(pairs))
    for p, (i, j) in enumerate(pairs):
        a, b = ids[offsets[i] : offsets[i + 1]], ids[offsets[j] : offsets[j + 1]]
        own_a, own_b = np.setdiff1d(a, b), np.setdiff1d(b, a)
        delta = np.median([points3D[k] for k in own_a], axis=0) - np.median([points3D[k] for k in own_b], axis=0)
        # The full set's extent stands in for the subset's: too few points to re-estimate it.
        cov = extent[i] / len(own_a) + extent[j] / len(own_b)
        d2[p] = delta @ np.linalg.solve(cov, delta)
    return d2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqs", nargs="*", help="default: the whole split")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--min-gap", type=int, default=30, help="frames between a pair's detections")
    ap.add_argument("--min-shared", type=int, default=2, help="3D points a pair must share")
    ap.add_argument("--min-frac", type=float, default=0.5, help="shared share of the smaller point set")
    ap.add_argument("--min-own", type=int, default=1, help="unshared points each side keeps")
    ap.add_argument("--min-pairs", type=int, default=50, help="fewer pairs leaves the gate null")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = data_root() / args.dataset / "landmarks"
    for seq in args.seqs or splits(args.dataset)[args.split]:
        lm = np.load(root / seq / "sfm.npz")
        pairs = linked_pairs(lm, args.min_gap, args.min_shared, args.min_frac, args.min_own)
        n_pairs = len(pairs)
        if n_pairs > MAX_PAIRS:
            pairs = pairs[rng.choice(n_pairs, MAX_PAIRS, replace=False)]

        points3D = read_points3D(data_root() / args.dataset / "recon" / seq / "model_txt" / "points3D.txt")
        median_d2 = float(np.median(mahalanobis2(lm, pairs, points3D))) if n_pairs else None
        k = median_d2 / chi2.median(3) if n_pairs >= args.min_pairs else None
        gate = k * chi2.ppf(args.quantile, 3) if k is not None else None
        result = {
            "gate": gate,
            "k": k,
            "median_d2": median_d2,
            "n_pairs": n_pairs,
            "n_dets_linked": int(len(np.unique(pairs))) if n_pairs else 0,
            "params": {
                "quantile": args.quantile,
                "min_gap": args.min_gap,
                "min_shared": args.min_shared,
                "min_frac": args.min_frac,
                "min_own": args.min_own,
                "min_pairs": args.min_pairs,
            },
        }
        (root / seq / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
        shown = f"{gate:8.1f}  k={k:6.2f}" if gate is not None else "    null"
        print(f"{args.dataset}/{seq:<16} gate={shown}  pairs={n_pairs}")


if __name__ == "__main__":
    main()
