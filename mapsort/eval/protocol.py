"""The frozen evaluation protocol: the `protocol` section of config/mapsort.yaml.

Kept separate from invocation: this says how results are scored, never which
ones. Dataset and sequence selection are CLI arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..data import DEFAULT_CONFIG


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Load the protocol, with the shape the evaluation functions expect."""
    resolved = Path(path) if path else DEFAULT_CONFIG
    protocol = yaml.safe_load(resolved.read_text())["protocol"]
    protocol["path"] = str(resolved)
    return protocol
