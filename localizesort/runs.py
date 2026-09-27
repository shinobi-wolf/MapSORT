"""Run directories: one per experiment, recording what was run and what it scored.

    runs/<date>_<module>/
      manifest.json    what was run: git state, config, sequences, timing
      metrics.json     what it scored; free-form, since each module reports
                       different things
      artifacts/       MOT files, plots, anything heavy

`manifest.json` and `metrics.json` are small and git-tracked, so a result stays
reproducible from its commit; `artifacts/` is ignored. Evaluation reads a run's
artifacts and never imports the pipeline, so any run can be re-scored later.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def git_state() -> dict[str, Any]:
    """Current commit and whether the tree was dirty, for reproducibility.

    A dirty tree means the commit alone does not identify what ran, so it is
    recorded rather than silently ignored.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain")),
    }


def create(
    module: str,
    *,
    task_id: str | None = None,
    config: dict[str, Any] | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Make a run directory and write its manifest. `task_id` is e.g. "T06"."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"{stamp}_{module}"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_dir.name,
        "module": module,
        "task_id": task_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "config": config or {},
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def read_yaml(path: Path) -> Any:
    """Read a config file, so the manifest can embed what the run resolved to."""
    import yaml

    return yaml.safe_load(Path(path).read_text())
