"""Cron gate: start the Claude supervisor only when there is something to do (see CLAUDE.md)."""

import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUTBOX = DATA / "recon_outbox.txt"
GSTATE = DATA / "recon_gate_state.json"
CLAUDE = Path.home() / ".local" / "bin" / "claude"
STALL_S = 3 * 3600
RETRIAGE_S = 6 * 3600
RENOTIFY_S = 3 * 3600
DIGEST_HOUR = 9
MAX_TURNS = {"notify": 8, "digest": 15, "triage": 60}


def log(msg: str) -> None:
    print(f"{datetime.now():%F %T}  {msg}", flush=True)


def attention() -> list[str]:
    seen, out = set(), []
    for p in sorted(DATA.glob("*/recon/*/STATE.json")):
        if p.resolve() in seen:
            continue
        seen.add(p.resolve())
        st = json.loads(p.read_text())
        key = f"{st['dataset']}/{st['seq']} {st['status']} attempt {st['attempt']}"
        if st["status"] in ("NEEDS_REVIEW", "FAILED"):
            out.append(key)
        elif st["status"] == "RUNNING":
            newest = max((f.stat().st_mtime for f in p.parent.glob("*.log")), default=p.stat().st_mtime)
            if time.time() - newest > STALL_S:
                out.append(f"{key} STALLED (no log change for {(time.time() - newest) / 3600:.1f} h)")
    return out


def main() -> None:
    lock = (DATA / "recon_gate.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    gs = json.loads(GSTATE.read_text()) if GSTATE.exists() else {}
    now, today = time.time(), datetime.now().strftime("%F")
    todo = attention()
    sig = hashlib.sha1("\n".join(todo).encode()).hexdigest()
    box = OUTBOX.read_text().strip() if OUTBOX.exists() else ""
    box_sig = hashlib.sha1(box.encode()).hexdigest()
    # An unchanged outbox means the last send failed; retry it slowly.
    notify_due = box and (box_sig != gs.get("notify_sig") or now - gs.get("notify_t", 0) > RENOTIFY_S)
    finished = (DATA / "recon_finished").exists()

    if todo and (sig != gs.get("triage_sig") or now - gs.get("triage_t", 0) > RETRIAGE_S):
        kind, detail = "triage", "Sequences needing attention:\n" + "\n".join(todo)
        gs.update(triage_sig=sig, triage_t=now)
    elif not finished and datetime.now().hour >= DIGEST_HOUR and gs.get("digest_day") != today:
        kind, detail = "digest", ""
        gs["digest_day"] = today
    elif notify_due:
        kind, detail = "notify", ""
    else:
        return

    gs.update(notify_sig=box_sig, notify_t=now)
    prompt = f"Run type: {kind}. Follow CLAUDE.md.\n{detail}".strip()
    cmd = [str(CLAUDE), "-p", prompt, "--add-dir", str(ROOT), "--permission-mode", "dontAsk",
           "--max-turns", str(MAX_TURNS[kind])]
    if kind == "triage":
        cmd += ["--model", "opus", "--settings", json.dumps({"effortLevel": "high"})]
    log(f"starting {kind}: {detail!r}")
    env = {"HOME": str(Path.home()), "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
    try:
        r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True, timeout=3600)
        log(f"{kind} exit {r.returncode}\n{r.stdout.strip()}\n{r.stderr.strip()}".rstrip())
    except subprocess.TimeoutExpired:
        log(f"{kind} timed out")
    GSTATE.write_text(json.dumps(gs, indent=2) + "\n")


if __name__ == "__main__":
    main()
