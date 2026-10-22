"""Box assignment, DLT triangulation and the orange sphere fit of Santos et al. (arXiv:2312.16724v1, Sec. 3.3)."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes a [N, 4] and b [M, 4]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def assign(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Eqs. 3-6: rectangular Jonker-Volgenant on 1 - IoU, then drop pairs with IoU = 0."""
    if not len(a) or not len(b):
        return []
    sim = iou(a, b)
    rows, cols = linear_sum_assignment(1.0 - sim)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if sim[r, c] > 0]


def camera_center(P: np.ndarray) -> np.ndarray:
    """Eq. 8: C = -M^-1 p4."""
    return -np.linalg.solve(P[:, :3], P[:, 3])


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Eq. 1 for one point X [3] in cameras P [N, 3, 4]: pixels [N, 2]."""
    x = P[:, :, :3] @ X + P[:, :, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return x[:, :2] / x[:, 2:3]


def dlt(c: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """Linear triangulation of one point from pixels c [N, 2] in cameras P [N, 3, 4]; None at infinity."""
    A = np.concatenate([c[:, :1] * P[:, 2] - P[:, 0], c[:, 1:] * P[:, 2] - P[:, 1]])
    X = np.linalg.svd(A)[2][-1]
    if not np.all(np.isfinite(X)) or abs(X[3]) <= 1e-12 * np.abs(X).max():
        return None
    return X[:3] / X[3]


def _samples(n: int, order: str, rng: np.random.Generator) -> Iterator[tuple[int, int, int]]:
    """Combinations(T, 3) of Algorithm 1, lexicographic or in random order without repeats."""
    if order == "lexicographic":
        yield from itertools.combinations(range(n), 3)
        return
    if comb(n, 3) <= 4096:
        combos = list(itertools.combinations(range(n), 3))
        for k in rng.permutation(len(combos)):
            yield combos[k]
        return
    seen: set[tuple[int, int, int]] = set()
    while len(seen) < comb(n, 3):
        s = tuple(sorted(rng.choice(n, 3, replace=False).tolist()))
        if s not in seen:
            seen.add(s)
            yield s


def ransac_triangulation(
    c: np.ndarray,
    P: np.ndarray,
    *,
    max_geom_error: float,
    inliers_ratio: float,
    max_iters: int,
    order: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Algorithm 1: orange centre X and the inlier mask over the track's boxes, or (None, None)."""
    n_iters = 1
    for s in _samples(len(c), order, rng):
        idx = list(s)
        X = dlt(c[idx], P[idx])
        if X is None:
            continue
        inliers = np.linalg.norm(project(P, X) - c, axis=1) <= max_geom_error
        if inliers.sum() / len(c) >= inliers_ratio:
            X = dlt(c[inliers], P[inliers])
            return (X, inliers) if X is not None else (None, None)
        n_iters += 1
        if n_iters > max_iters:
            return None, None
    return None, None


def estimate_orange(
    boxes: np.ndarray, P: np.ndarray, C: np.ndarray, focal: np.ndarray, *, c_ray: float, **ransac
) -> tuple[np.ndarray | None, float | None]:
    """Algorithm 2: sphere centre and radius from xyxy boxes [N, 4] seen by cameras P, centres C, focals."""
    # Algorithm 1 line 4 prints ((x + width) / 2, ...), a typo for the centroid: relocalization only works with it.
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    X, inliers = ransac_triangulation(centers, P, **ransac)
    if X is None:
        return None, None
    r = np.maximum(boxes[inliers, 2] - boxes[inliers, 0], boxes[inliers, 3] - boxes[inliers, 1]) / 2
    d = np.linalg.norm(X - C[inliers], axis=1)
    return X, c_ray * float(np.median(d * r / focal[inliers]))


def reprojected_box(X: np.ndarray, r3d: float, P: np.ndarray, C: np.ndarray, focal: float) -> np.ndarray:
    """Sec. 3.3.3: the square xyxy box of the sphere (X, r3d) seen by camera P."""
    x = project(P[None], X)[0]
    r = r3d * focal / np.linalg.norm(X - C)
    return np.array([x[0] - r, x[1] - r, x[0] + r, x[1] + r])
