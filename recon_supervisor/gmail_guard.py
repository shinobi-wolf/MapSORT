"""PreToolUse hook: let the recon supervisor email only the project owner."""
import json
import sys

OWNER = "vemshari27@gmail.com"

inp = json.load(sys.stdin).get("tool_input", {})
to = [a.strip().lower() for a in inp.get("to") or []]
bad = (to != [OWNER] or inp.get("cc") or inp.get("bcc") or inp.get("draftId")
       or inp.get("attachments"))
if bad:
    print(f"blocked: supervisor may only send plain mail to {OWNER}", file=sys.stderr)
    sys.exit(2)
