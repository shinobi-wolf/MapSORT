"""The frozen evaluation protocol, from config/eval.yaml.

Kept separate from invocation: this says how results are scored, never which
ones. Dataset and sequence selection are CLI arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "config" / "eval.yaml"


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Load the protocol, with the shape the evaluation functions expect."""
    resolved = Path(path) if path else DEFAULT_PROTOCOL
    protocol = yaml.safe_load(resolved.read_text())
    protocol["path"] = str(resolved)
    return protocol
