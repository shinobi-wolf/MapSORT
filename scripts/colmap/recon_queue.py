"""Run config/recon_plan.yaml one sequence at a time, resumably; exits when all are DONE or BLOCKED.

State: data/<dataset>/recon/<seq>/STATE.json. Events: data/recon_outbox.txt (emailed by
recon_supervisor/). Summary: data/recon_status.txt. Lifecycle: recon_supervisor/CLAUDE.md.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PLAN = ROOT / "config" / "recon_plan.yaml"
REGISTRY = ROOT / "config" / "datasets.yaml"
SCRIPT = ROOT / "scripts" / "colmap" / "sparse_recon.sh"
OUTBOX = DATA / "recon_outbox.txt"
STATUS = DATA / "recon_status.txt"
FINISHED = DATA / "recon_finished"
MIN_RATIO, MAX_REPROJ = 0.9, 1.5
TERMINAL = {"DONE", "BLOCKED"}
POLL_S = 300
OUTPUTS = ["sparse", "refined", "model_txt", "refined.ply"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def outbox(msg: str) -> None:
    with OUTBOX.open("a") as f:
        f.write(f"{now()}  {msg}\n")
    print(f"{now()}  {msg}", flush=True)


class Seq:
    def __init__(self, entry: dict, defaults: dict):
        self.dataset, self.split, self.name = entry["dataset"], entry["split"], str(entry["seq"])
        self.key = f"{self.dataset}/{self.name}"
        self.dir = DATA / self.dataset / "recon" / self.name
        self.images = DATA / self.dataset / "mot" / self.split / self.name / "img1"
        self.alias = entry.get("alias")
        self.env = {k: str(v) for k, v in {**defaults, **entry.get("env", {})}.items()}
        self.intrinsics = entry.get("intrinsics", "free")
        self.dep = (f"{self.dataset}/{self.alias}" if self.alias
                    else self.intrinsics.get("from") if isinstance(self.intrinsics, dict) else None)

    def load(self) -> dict:
        p = self.dir / "STATE.json"
        if not self.alias and p.exists():
            return json.loads(p.read_text())
        return {"dataset": self.dataset, "split": self.split, "seq": self.name,
                "status": "PENDING", "attempt": 0, "history": []}

    def save(self, st: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "STATE.json.tmp"
        tmp.write_text(json.dumps(st, indent=2) + "\n")
        os.replace(tmp, self.dir / "STATE.json")


def load_plan() -> list[Seq]:
    plan = yaml.safe_load(PLAN.read_text())
    return [Seq(e, plan.get("defaults", {})) for e in plan["sequences"]]


def status(s: Seq, by_key: dict[str, Seq]) -> str:
    if not s.alias:
        return s.load()["status"]
    target = by_key[s.dep].load()["status"]
    if target == "DONE":
        return "DONE" if s.dir.is_symlink() else "PENDING"
    return "BLOCKED" if target == "BLOCKED" else "PENDING"


def analyze(model: Path) -> dict:
    out = subprocess.run(["colmap", "model_analyzer", "--path", str(model)], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    reg = re.search(r"Registered images: (\d+)", out)
    err = re.search(r"Mean reprojection error: ([\d.]+)px", out)
    return {"registered": int(reg.group(1)) if reg else 0, "reproj": float(err.group(1)) if err else None}


def evaluate(s: Seq) -> dict:
    models = {m.name: analyze(m)["registered"] for m in sorted((s.dir / "sparse").iterdir()) if m.is_dir()}
    refined = s.dir / "refined"
    final = refined if (refined / "images.bin").exists() else (
        s.dir / "sparse" / max(models, key=models.get) if models else None)
    res = analyze(final) if final else {"registered": 0, "reproj": None}
    frames = len(list(s.images.iterdir()))
    cams = s.dir / "model_txt" / "cameras.txt"
    camera = next((l.strip() for l in cams.read_text().splitlines() if l and l[0] != "#"), None) \
        if cams.exists() else None
    return {**res, "frames": frames, "ratio": round(res["registered"] / frames, 3),
            "n_models": len(models), "models": models, "camera": camera}


def camera_from(s: Seq, dep: Seq) -> str:
    line = dep.load()["result"]["camera"].split()
    w, h = int(line[2]), int(line[3])
    info = (s.images.parent / "seqinfo.ini").read_text()
    sw, sh = (int(re.search(rf"{k}=(\d+)", info).group(1)) for k in ("imWidth", "imHeight"))
    if (w, h) != (sw, sh):
        raise ValueError(f"{dep.key} camera is {w}x{h}, {s.key} images are {sw}x{sh}")
    return ",".join(line[4:])


def read_override(s: Seq) -> dict:
    p = s.dir / "override.env"
    if not p.exists():
        return {}
    pairs = (l.split("=", 1) for l in p.read_text().splitlines() if "=" in l and not l.startswith("#"))
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def archive(s: Seq, name: str, with_db: bool) -> None:
    dest = s.dir / name
    dest.mkdir(exist_ok=True)
    moving = [s.dir / o for o in OUTPUTS] + list(s.dir.glob("0*.log"))
    if with_db:
        moving += list(s.dir.glob("database.db*"))
    for p in moving:
        if p.exists():
            shutil.move(str(p), dest / p.name)


def register_recon(dataset: str, seq: str) -> None:
    """Add seq to the dataset's `recon:` list in datasets.yaml, keeping the file's comments."""
    lines = REGISTRY.read_text().splitlines(keepends=True)
    inside = False
    for i, line in enumerate(lines):
        if re.match(rf"^{dataset}:\s*$", line):
            inside = True
        elif inside and re.match(r"^\S", line):
            break
        elif inside and re.match(r"^  recon:\s*\[", line):
            items = yaml.safe_load(line.split(":", 1)[1])
            if seq not in items:
                lines[i] = "  recon: " + json.dumps(sorted(items + [seq])) + "\n"
                REGISTRY.write_text("".join(lines))
            return


def run(s: Seq, by_key: dict[str, Seq], resume: bool) -> None:
    st = s.load()
    if resume:
        params = dict(st["params"])
        params["START"] = ("map" if (s.dir / "03_mapper.log").exists()
                           else "match" if (s.dir / "02_match.log").exists() else "extract")
        if params["START"] == "map":
            archive(s, f"crash_{datetime.now():%Y%m%d_%H%M%S}", with_db=False)
        outbox(f"{s.key} resuming attempt {st['attempt']} from {params['START']} after driver restart")
    else:
        override = read_override(s)
        st["attempt"] += 1
        params = dict(s.env)
        if s.intrinsics == "free":
            params["CAMERA_PARAMS"] = ""
        elif isinstance(s.intrinsics, str):
            params["CAMERA_PARAMS"] = s.intrinsics
        elif by_key[s.dep].load()["status"] == "DONE":
            params["CAMERA_PARAMS"] = camera_from(s, by_key[s.dep])
        else:
            params["CAMERA_PARAMS"] = ""
            outbox(f"{s.key} using free intrinsics: {s.dep} is BLOCKED")
        params.update(override)
        params.setdefault("START", "extract")
        if st["attempt"] > 1:
            archive(s, f"attempt_{st['attempt'] - 1}", with_db=params["START"] == "extract")

    st.update(status="RUNNING", started=now(), finished=None, params=params)
    s.save(st)
    write_status(list(by_key.values()), by_key)
    print(f"{now()}  {s.key} attempt {st['attempt']} START={params['START']} {params}", flush=True)
    t0 = time.time()
    with (s.dir / "run.log").open("a") as log:
        rc = subprocess.run(["bash", str(SCRIPT), str(s.images), str(s.dir)],
                            env={**os.environ, **params}, stdout=log, stderr=subprocess.STDOUT).returncode
    res = evaluate(s)
    ok = rc == 0 and res["ratio"] >= MIN_RATIO and res["reproj"] is not None and res["reproj"] < MAX_REPROJ
    new = "DONE" if ok else "FAILED" if rc != 0 else "NEEDS_REVIEW"
    hours = (time.time() - t0) / 3600
    st = s.load()
    st.update(status=new, finished=now(), exit_code=rc, result=res)
    st["history"].append({"attempt": st["attempt"], "params": params, "exit_code": rc, "result": res,
                          "status": new, "started": st["started"], "finished": st["finished"]})
    s.save(st)
    outbox(f"{s.key} {new} attempt {st['attempt']} ({hours:.1f} h, exit {rc}): "
           f"{res['registered']}/{res['frames']} registered, {res['n_models']} models, "
           f"reproj {res['reproj']} px")
    if new == "DONE":
        register_recon(s.dataset, s.name)


def write_status(seqs: list[Seq], by_key: dict[str, Seq]) -> None:
    rows = [f"updated {now()}", ""]
    for s in seqs:
        if s.alias:
            rows.append(f"{s.key:28} {status(s, by_key):13} alias of {s.alias}")
            continue
        st = s.load()
        r = st.get("result") or {}
        rows.append(f"{s.key:28} {st['status']:13} attempt {st['attempt']}  "
                    + (f"{r['registered']}/{r['frames']} reg, {r['n_models']} models, reproj {r['reproj']}"
                       if r else "")
                    + (f"  since {st['started']}" if st["status"] == "RUNNING" else ""))
    STATUS.write_text("\n".join(rows) + "\n")


def main() -> None:
    lock = (DATA / "recon_queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another recon_queue is running")

    seqs = load_plan()
    by_key = {s.key: s for s in seqs}
    for s in seqs:
        if not s.alias and s.load()["status"] == "RUNNING":
            run(s, by_key, resume=True)

    while True:
        seqs = load_plan()
        by_key = {s.key: s for s in seqs}
        for s in seqs:
            if s.alias and status(s, by_key) == "PENDING" and by_key[s.dep].load()["status"] == "DONE":
                s.dir.symlink_to(s.alias)
                register_recon(s.dataset, s.name)
                outbox(f"{s.key} DONE: linked to {s.alias}")
        write_status(seqs, by_key)

        if all(status(s, by_key) in TERMINAL for s in seqs):
            outbox("ALL FINISHED")
            FINISHED.touch()
            return
        nxt = next((s for s in seqs if not s.alias and s.load()["status"] == "PENDING"
                    and (s.dep is None or by_key[s.dep].load()["status"] in TERMINAL)), None)
        if nxt is None:
            time.sleep(POLL_S)
            continue
        try:
            run(nxt, by_key, resume=False)
        except Exception as e:
            st = nxt.load()
            st.update(status="FAILED", finished=now(), error=repr(e))
            nxt.save(st)
            outbox(f"{nxt.key} FAILED in driver: {e!r}")


if __name__ == "__main__":
    main()
