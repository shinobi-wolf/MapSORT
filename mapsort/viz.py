"""Track overlay rendering: boxes, identities, and an optional ground-truth outline.

Boxes are coloured either by match status -- black where a prediction corresponds
to a ground-truth object, red where it does not -- or by track id, where an
identity keeps its colour for as long as the tracker keeps calling it the same
thing, so an ID switch shows up as a box changing colour. That second mode is
the one that makes this project's failure visible.
"""

from __future__ import annotations

import os

# Opening a window in this environment emits ~130 `QObject::moveToThread`
# warnings from a Qt plugin conflict elsewhere in the env. The windows work
# fine, so silence the noise rather than chase it. These come through Qt's
# default logging category, which is why the rule targets `default.warning`
# specifically instead of a qt.qpa.* rule -- other categories still report.
# Must be set before cv2 imports Qt.
os.environ.setdefault("QT_LOGGING_RULES", "default.warning=false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .schemas import Detection  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Matching the sizing of MapSORT's own comparison figures, so overlays from
# the two projects can be read side by side.
FONT_SCALE = 1.0
BOX_THICKNESS = 3
LABEL_PAD = 3  # padding inside the filled label chip above each box

MATCH_COLOUR = (0, 0, 0)  # BGR black: prediction matched to a GT object
MISMATCH_COLOUR = (0, 0, 255)  # BGR red: prediction matching no GT object
MATCH_IOU = 0.5

HEADER_HEIGHT = 55
HEADER_BG = (20, 20, 20)
HEADER_FONT_SCALE, HEADER_THICKNESS = 1.4, 3


def id_colour(track_id: int) -> tuple[int, int, int]:
    """A stable, well-separated BGR colour for a track id.

    The golden-ratio hue step keeps consecutive ids far apart in hue, so
    neighbouring objects do not come out nearly the same colour.
    """
    hue = int((track_id * 0.618033988749895 * 180) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def _iou_matrix(a: list[Detection], b: list[Detection]) -> np.ndarray:
    """Pairwise IoU between two lists of xyxy detections."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    boxes_a = np.array([d.bbox for d in a], dtype=float)
    boxes_b = np.array([d.bbox for d in b], dtype=float)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area = lambda box: (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])  # noqa: E731
    union = area(boxes_a)[:, None] + area(boxes_b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def matched_mask(
    detections: list[Detection],
    gt_detections: list[Detection],
    iou: float = MATCH_IOU,
) -> list[bool]:
    """Which predictions correspond to a ground-truth object in this frame.

    One-to-one Hungarian assignment on IoU, the same rule CLEAR-MOT uses, so a
    box drawn red here is one the metrics also count as a false positive.
    """
    from scipy.optimize import linear_sum_assignment

    matched = [False] * len(detections)
    similarity = _iou_matrix(detections, gt_detections)
    if similarity.size == 0:
        return matched

    rows, cols = linear_sum_assignment(-similarity)
    for row, col in zip(rows, cols):
        if similarity[row, col] >= iou:
            matched[row] = True
    return matched


def _label(canvas: np.ndarray, text: str, x: int, y: int, colour: tuple) -> None:
    """Draw a label in a filled chip above `y`, white text on the box's own colour."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, BOX_THICKNESS)
    top = max(0, y - th - 10)
    cv2.rectangle(canvas, (x, top), (x + tw + 2 * LABEL_PAD, y), colour, -1)
    cv2.putText(
        canvas,
        text,
        (x + LABEL_PAD, max(th, y - 6)),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        BOX_THICKNESS,
        cv2.LINE_AA,
    )


def draw_tracks(
    frame: np.ndarray,
    detections: list[Detection],
    track_ids: list[int],
    *,
    gt_detections: list[Detection] | None = None,
    colour_by: str = "match",
    label: str = "",
) -> np.ndarray:
    """Annotate an RGB frame and return it as BGR, ready for `cv2.imshow`.

    `colour_by` picks what colour means:

    "match"  black where the prediction corresponds to a ground-truth object,
             red where it does not -- the MapSORT comparison-figure look.
             Needs `gt_detections`; falls back to "id" without them.
    "id"     a stable colour per track id, so an ID switch shows up as a box
             changing colour.
    """
    canvas = np.ascontiguousarray(frame[..., ::-1]).copy()

    matched = None
    if colour_by == "match" and gt_detections is not None:
        matched = matched_mask(detections, gt_detections)

    for i, (det, track_id) in enumerate(zip(detections, track_ids)):
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        if matched is None:
            colour = id_colour(track_id)
        else:
            colour = MATCH_COLOUR if matched[i] else MISMATCH_COLOUR

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, BOX_THICKNESS)
        _label(canvas, f"id={track_id}", x1, y1, colour)

    if label:
        cv2.putText(canvas, label, (10, 32), FONT, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, label, (10, 32), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def draw_gt(frame: np.ndarray, gt_detections: list[Detection], gt_ids: list[int]) -> np.ndarray:
    """The ground-truth panel: every object black, labelled with its GT id."""
    canvas = np.ascontiguousarray(frame[..., ::-1]).copy()
    for det, gt_id in zip(gt_detections, gt_ids):
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), MATCH_COLOUR, BOX_THICKNESS)
        _label(canvas, f"id={gt_id}", x1, y1, MATCH_COLOUR)
    return canvas


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    """A titled strip above a panel, so stacked panels stay identifiable."""
    width = image.shape[1]
    header = np.full((HEADER_HEIGHT, width, 3), HEADER_BG, dtype=np.uint8)

    # Shrink to fit rather than run off the edge: a narrow panel with a long
    # title would otherwise lose the end of it.
    scale, thickness = HEADER_FONT_SCALE, HEADER_THICKNESS
    while scale > 0.35:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
        if tw <= width - 12:
            break
        scale -= 0.1
        thickness = max(1, int(round(scale * 2)))
    cv2.putText(
        header,
        text,
        (max(0, (width - tw) // 2), (HEADER_HEIGHT + th) // 2),
        FONT,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def side_by_side(panels: list[tuple[np.ndarray, str]], gap: int = 6) -> np.ndarray:
    """Lay titled panels out left to right, separated by a dark gutter."""
    titled = [add_header(image, title) for image, title in panels]
    height = max(p.shape[0] for p in titled)
    padded = [
        np.vstack([p, np.full((height - p.shape[0], p.shape[1], 3), HEADER_BG, np.uint8)])
        if p.shape[0] < height
        else p
        for p in titled
    ]

    separator = np.full((height, gap, 3), HEADER_BG, dtype=np.uint8)
    out: list[np.ndarray] = []
    for i, panel in enumerate(padded):
        if i:
            out.append(separator)
        out.append(panel)
    return np.hstack(out)


class Window:
    """An imshow window that can be paused, stepped and quit.

    Used as a context manager so the window is destroyed even if the run raises.
    `q` or Esc requests a stop, space toggles pause; while paused, any key steps
    one frame.
    """

    def __init__(self, title: str = "mapsort", height: int = 1000) -> None:
        self.title = title
        self.height = height
        self.paused = False
        self.stopped = False
        self._sized = False

    @staticmethod
    def available() -> tuple[bool, str]:
        """Whether a window can actually be opened, and why not if it cannot.

        Probed up front: a silently ignored `--viz` over SSH is worse than an
        error, and OpenCV built headless raises only when first asked to draw.
        """
        try:
            cv2.namedWindow("__mapsort_probe__")
            cv2.destroyWindow("__mapsort_probe__")
            cv2.waitKey(1)
            return True, ""
        except cv2.error as exc:
            return False, str(exc).strip().splitlines()[-1] if str(exc).strip() else "cv2 GUI unavailable"

    def __enter__(self) -> "Window":
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        return self

    def __exit__(self, *exc) -> None:
        cv2.destroyAllWindows()
        # OpenCV needs a few event-loop turns to actually tear the window down.
        for _ in range(4):
            cv2.waitKey(1)

    def show(self, canvas: np.ndarray) -> bool:
        """Display one frame. Returns False once a stop has been requested."""
        if self.stopped:
            return False

        # WINDOW_NORMAL opens at an arbitrary small size unless told otherwise,
        # so size it once from the first frame's aspect ratio.
        if not self._sized:
            h, w = canvas.shape[:2]
            cv2.resizeWindow(self.title, int(w * self.height / h), self.height)
            self._sized = True

        cv2.imshow(self.title, canvas)
        while True:
            key = cv2.waitKey(0 if self.paused else 1) & 0xFF
            if key in (ord("q"), 27):
                self.stopped = True
                return False
            if key == ord(" "):
                self.paused = not self.paused
                if not self.paused:
                    break
                continue
            if self.paused:
                break  # any other key steps a single frame
            break
        return True


def writer(path, frame_shape: tuple[int, int], fps: float) -> cv2.VideoWriter:
    """An mp4 writer for the overlay, for when there is no display to show it on."""
    height, width = frame_shape
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return out
