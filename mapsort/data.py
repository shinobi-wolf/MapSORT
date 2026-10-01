"""Dataset access, driven by config/datasets.yaml. Layout: data/README.md.

    seq = sequence("lettucemots", "test", "0011")
    frames = seq.load_frames(limit=60)
    dets = seq.gt_detections()          # frame index -> list[Detection]
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .schemas import Detection, LandmarkObservation

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "config" / "datasets.yaml"
DEFAULT_CONFIG = REPO_ROOT / "config" / "mapsort.yaml"


def data_root() -> Path:
    env = os.environ.get("MAPSORT_DATA")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


@lru_cache(maxsize=None)
def registry(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_REGISTRY).read_text())


@dataclass(frozen=True)
class Sequence:
    dataset: str
    split: str
    name: str
    path: Path  # mot/<split>/<seq>
    dets_path: Path | None  # dets/<seq>.txt
    landmark_dir: Path  # landmarks/<seq>/
    landmark_backends: tuple[str, ...]  # backends with landmarks/<seq>/<backend>.txt; empty if none
    recon_path: Path | None  # recon/<seq>/

    @property
    def image_dir(self) -> Path:
        return self.path / "img1"

    @property
    def gt_path(self) -> Path:
        return self.path / "gt" / "gt.txt"

    @property
    def info(self) -> dict[str, str]:
        """Parsed seqinfo.ini `[Sequence]`."""
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(self.path / "seqinfo.ini")
        return dict(cfg["Sequence"])

    @property
    def num_frames(self) -> int:
        return int(self.info["seqLength"])

    def landmark_path(self, backend: str) -> Path:
        """`landmarks/<seq>/<backend>.npz`, e.g. backend `sfm` for the RLoc pseudo-GT."""
        if backend not in self.landmark_backends:
            raise FileNotFoundError(
                f"{self.dataset}/{self.name} has no {backend!r} landmarks "
                f"(has: {list(self.landmark_backends)}); list it under `landmarks` in "
                f"config/datasets.yaml and build them (see data/README.md)"
            )
        return self.landmark_dir / f"{backend}.npz"

    def image_paths(self) -> list[Path]:
        """Frame paths in order; index 0 is frame 1."""
        return sorted(self.image_dir.glob(f"*{self.info['imExt']}"))

    def iter_frames(self, limit: int | None = None):
        """Yield RGB uint8 frames. Stream in runs; a full sequence is ~640 MB."""
        import cv2

        for path in self.image_paths()[:limit]:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"could not read frame {path}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_frames(self, limit: int | None = None) -> list[np.ndarray]:
        """All frames at once, for notebooks."""
        return list(self.iter_frames(limit))

    def gt_rows(self) -> np.ndarray:
        """Raw `gt.txt` as an (N, 9) array."""
        return np.atleast_2d(np.loadtxt(self.gt_path, delimiter=","))

    def stored_detections(self, limit: int | None = None) -> list[list[Detection]]:
        """`dets/<seq>.txt` as `Detection`s, indexed by 0-based frame."""
        if self.dets_path is None:
            raise FileNotFoundError(
                f"{self.dataset}/{self.name} has no stored detections; "
                f"list it under `dets` in config/datasets.yaml and build them "
                f"(see data/README.md)"
            )
        num = self.num_frames if limit is None else min(limit, self.num_frames)
        dets: list[list[Detection]] = [[] for _ in range(num)]

        for frame_no, _id, x, y, w, h, conf, class_id in np.atleast_2d(
            np.loadtxt(self.dets_path, delimiter=",")
        ):
            idx = int(frame_no) - 1
            if idx >= num:
                continue
            dets[idx].append(
                Detection(
                    bbox=np.array([x, y, x + w, y + h], dtype=float),
                    confidence=float(conf),
                    class_id=int(class_id),
                )
            )
        return dets

    def stored_landmarks(
        self, backend: str, limit: int | None = None
    ) -> list[list[LandmarkObservation | None]]:
        """`landmarks/<seq>/<backend>.npz`, indexed by 0-based frame.

        Each frame's list is parallel to that frame's `stored_detections`, with
        `None` where the detection yielded no landmark.
        """
        data = np.load(self.landmark_path(backend))
        num = self.num_frames if limit is None else min(limit, self.num_frames)
        out: list[list[LandmarkObservation | None]] = [[] for _ in range(num)]

        for frame_no, has, position, covariance, extent in zip(
            data["frame"], data["has_landmark"], data["position"],
            data["covariance"], data["extent"]
        ):
            idx = int(frame_no) - 1
            if idx >= num:
                continue
            out[idx].append(
                LandmarkObservation(
                    position=position.astype(float),
                    covariance=covariance.astype(float),
                    extent=extent.astype(float),
                )
                if has
                else None
            )
        return out

    def gt_detections(
        self, limit: int | None = None
    ) -> tuple[list[list[Detection]], list[list[int]]]:
        """GT as `Detection`s plus track ids, indexed by 0-based frame."""
        num = self.num_frames if limit is None else min(limit, self.num_frames)
        dets: list[list[Detection]] = [[] for _ in range(num)]
        ids: list[list[int]] = [[] for _ in range(num)]

        for frame_no, track_id, x, y, w, h, _conf, class_id, _vis in self.gt_rows():
            idx = int(frame_no) - 1
            if idx >= num:
                continue
            dets[idx].append(
                Detection(
                    bbox=np.array([x, y, x + w, y + h], dtype=float),
                    confidence=1.0,
                    class_id=int(class_id),
                )
            )
            ids[idx].append(int(track_id))
        return dets, ids


def dataset_spec(dataset: str) -> dict:
    spec = registry().get(dataset)
    if spec is None:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(registry())}")
    return spec


def splits(dataset: str) -> dict[str, list[str]]:
    return dataset_spec(dataset)["splits"]


def sequence(dataset: str, split: str, name: str) -> Sequence:
    spec = dataset_spec(dataset)
    if name not in spec["splits"].get(split, []):
        raise KeyError(f"{dataset} has no sequence {name!r} in split {split!r}")
    root = data_root() / dataset

    path = root / "mot" / split / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"{dataset}/{split}/{name} is missing at {path}. "
            "Build the MOT view first -- see data/README.md."
        )

    def listed(kind: str, rel: str) -> Path | None:
        return root / kind / rel if name in spec.get(kind, []) else None

    return Sequence(
        dataset=dataset,
        split=split,
        name=name,
        path=path,
        dets_path=listed("dets", f"{name}.txt"),
        landmark_dir=root / "landmarks" / name,
        landmark_backends=tuple(spec.get("landmarks", {}).get(name, ())),
        recon_path=listed("recon", name),
    )


def sequences(dataset: str, split: str, names: list[str] | None = None) -> list[Sequence]:
    """Every sequence in a split, or just `names`."""
    available = splits(dataset)
    if split not in available:
        raise KeyError(f"{dataset} has no split {split!r}; known: {sorted(available)}")
    return [sequence(dataset, split, name) for name in (names or available[split])]
