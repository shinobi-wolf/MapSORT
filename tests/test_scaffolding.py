import numpy as np
import pytest

from localizesort import LocalizeSORT
from localizesort.schemas import RGBFrame


def test_pipeline_constructs():
    LocalizeSORT()


def test_pipeline_wiring_reaches_detector_stub():
    pipeline = LocalizeSORT()
    frame = RGBFrame(image=np.zeros((4, 4, 3), dtype=np.uint8), frame_index=0)

    # Detector is the first stub in the wiring, so it should be the one to
    # raise until its internals are implemented.
    with pytest.raises(NotImplementedError):
        pipeline(frame)
