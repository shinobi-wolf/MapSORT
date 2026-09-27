"""Dataset access, driven by config/datasets.yaml.

Every dataset is exposed through the same canonical MOT layout, so nothing
downstream branches on which dataset it is running against:

    <data_root>/<dataset>/mot/<split>/<seq>/{img1,gt/gt.txt,seqinfo.ini}
    <data_root>/<dataset>/recon/<seq>/

The data root is `$LOCALIZESORT_DATA` if set, else the repo's `data/`. See
data/README.md for how the tree is built.

Typical use:

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

from .schemas import Detection

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "config" / "datasets.yaml"


def data_root() -> Path:
    """The dataset root: `$LOCALIZESORT_DATA`, else the repo's `data/`."""
    env = os.environ.get("LOCALIZESORT_DATA")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


@lru_cache(maxsize=None)
def registry(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_REGISTRY).read_text())


@dataclass(frozen=True)
class Sequence:
    """One sequence's canonical MOT directory, plus its optional reconstruction."""

    dataset: str
    split: str
    name: str
    path: Path  # the mot/<split>/<seq> directory
    recon_path: Path | None  # offline SfM reconstruction, when one exists

    @property
    def image_dir(self) -> Path:
        return self.path / "img1"

    @property
    def gt_path(self) -> Path:
        return self.path / "gt" / "gt.txt"

    @property
    def info(self) -> dict[str, str]:
        """Parsed seqinfo.ini `[Sequence]` section."""
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(self.path / "seqinfo.ini")
        return dict(cfg["Sequence"])

    @property
    def num_frames(self) -> int:
        return int(self.info["seqLength"])

    def image_paths(self) -> list[Path]:
        """Frame paths in order. Filenames are 1-based; index 0 is frame 1."""
        return sorted(self.image_dir.glob(f"*{self.info['imExt']}"))

    def iter_frames(self, limit: int | None = None):
        """Yield frames as RGB uint8 arrays, which is what the pipeline expects.

        A 245-frame 810x1080 sequence is ~640 MB decoded, so anything running the
        whole pipeline should stream rather than use `load_frames`.
        """
        import cv2  # local: only needed when frames are actually loaded

        for path in self.image_paths()[:limit]:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"could not read frame {path}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_frames(self, limit: int | None = None) -> list[np.ndarray]:
        """All frames at once. Convenient in notebooks; use `iter_frames` in runs."""
        return list(self.iter_frames(limit))

    def gt_rows(self) -> np.ndarray:
        """Raw `gt.txt` as an (N, 9) array, in MOT's own conventions."""
        return np.atleast_2d(np.loadtxt(self.gt_path, delimiter=","))

    def gt_detections(
        self, limit: int | None = None
    ) -> tuple[list[list[Detection]], list[list[int]]]:
        """Ground truth as pipeline `Detection`s, plus the matching track ids.

        Both lists are indexed by 0-based frame index, so they line up with
        `load_frames`. `gt.txt` frames are 1-based and its boxes are top-left
        `xywh`, while `Detection.bbox` is `xyxy`.
        """
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
    root = data_root() / spec["root"]

    path = root / spec["mot"].format(split=split, seq=name)
    if not path.is_dir():
        raise FileNotFoundError(
            f"{dataset}/{split}/{name} is missing at {path}. "
            "Build the MOT view first -- see data/README.md."
        )

    recon = root / spec["recon"].format(seq=name)
    return Sequence(
        dataset=dataset,
        split=split,
        name=name,
        path=path,
        recon_path=recon if recon.exists() else None,
    )


def sequences(dataset: str, split: str, names: list[str] | None = None) -> list[Sequence]:
    """Every sequence in a split, or just `names`."""
    available = splits(dataset)
    if split not in available:
        raise KeyError(f"{dataset} has no split {split!r}; known: {sorted(available)}")
    return [sequence(dataset, split, name) for name in (names or available[split])]
