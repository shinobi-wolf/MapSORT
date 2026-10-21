"""Online camera resection: the frame's 3x4 projection, fitted to tracked landmarks and their boxes."""

from __future__ import annotations

import numpy as np


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: homogeneous points and the similarity that produced them."""
    c = pts.mean(0)
    s = np.sqrt(pts.shape[1]) / max(np.mean(np.linalg.norm(pts - c, axis=1)), 1e-12)
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return np.c_[pts, np.ones(len(pts))] @ T.T, T


def dlt(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """3x4 projection from >= 6 3D-2D correspondences (direct linear transform)."""
    Xn, TX = _normalize(X)
    xn, Tx = _normalize(x)
    u, v = xn[:, :1] / xn[:, 2:], xn[:, 1:2] / xn[:, 2:]
    z = np.zeros_like(Xn)
    A = np.vstack([np.hstack([z, -Xn, v * Xn]), np.hstack([Xn, z, -u * Xn])])
    return np.linalg.inv(Tx) @ np.linalg.svd(A)[2][-1].reshape(3, 4) @ TX


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pixel coordinates of map points X [N, 3]; nan behind the camera's plane at infinity."""
    h = np.c_[X, np.ones(len(X))] @ P.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :2] / h[:, 2:3]


def _inliers(P: np.ndarray, X: np.ndarray, x: np.ndarray, scale: np.ndarray, thr: float) -> np.ndarray:
    err = np.linalg.norm(project(P, X) - x, axis=1) / scale
    return np.nan_to_num(err, nan=np.inf) < thr


def resect(
    X: np.ndarray,
    x: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
    *,
    iters: int = 300,
    inlier: float = 0.5,
    min_inliers: int = 8,
    prior: np.ndarray | None = None,
    refits: int = 5,
) -> np.ndarray | None:
    """RANSAC DLT; errors are in units of `scale` (box sizes). None when too few points agree.

    The previous frame's camera competes with the random hypotheses, and the winner is refit on its
    inliers until they settle, which keeps the estimate stable from frame to frame.
    """
    if len(X) < max(6, min_inliers):
        return None
    best = _inliers(prior, X, x, scale, inlier) if prior is not None else None
    for _ in range(iters):
        idx = rng.choice(len(X), 6, replace=False)
        try:
            inl = _inliers(dlt(X[idx], x[idx]), X, x, scale, inlier)
        except np.linalg.LinAlgError:
            continue
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_inliers:
        return None
    P = None
    for _ in range(refits):
        try:
            P = dlt(X[best], x[best])
        except np.linalg.LinAlgError:
            return P
        new = _inliers(P, X, x, scale, inlier)
        if new.sum() < min_inliers or np.array_equal(new, best):
            break
        best = new
    return P
