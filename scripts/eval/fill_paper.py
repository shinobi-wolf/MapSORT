"""Copy docs/precision-ag-paper/results/*.csv into the result tables and plots of sn-article.tex.

    python scripts/eval/fill_paper.py

Only table cells and plot coordinates change; prose and captions are left alone. Missing values
stay \\tbd. Best values are bolded only within a complete group (all trackers or variants present),
so a partial table carries no misleading bold.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "docs" / "precision-ag-paper"
TEX = PAPER / "sn-article.tex"
RESULTS = PAPER / "results"

METRICS = ["HOTA", "AssA", "DetA", "MOTA", "IDF1", "IDSW", "CE", "RIP"]
TRACKERS = ["BoT-SORT", "SAM2MOT", "StrongSORT", "MapSORT"]
SEQ_LABELS = {  # tab_sequences / tab_runtime / fig_gate sequence -> label used in the tex rows
    "LettuceMOTS 0011": "LettuceMOTS 0011",
    "MOrangeT V01": "MOrangeT V01",
    "GrapeMOTS PathPlanning_7": r"GrapeMOTS PP\_7",
    "AppleMOTS 0008": "AppleMOTS 0008",
}
GATE_SERIES = {  # fig_gate sequence -> pgfplots style of its series
    "LettuceMOTS 0011": "clm!75!black, mark=*",
    "MOrangeT V01": "capp, mark=square*",
    "GrapeMOTS PathPlanning_7": "cmap, mark=triangle*",
    "AppleMOTS 0008": "black!60, mark=diamond*",
}
GAP_STYLE = {
    "StrongSORT": "fill=black!35",
    "MapSORT": "fill=clm!75!black",
    "BoT-SORT": "fill=black!12, draw=black!50, dashed",
    "SAM2MOT": "fill=black!12, draw=black!50, dotted",
}
GAP_BINS = {"10-29": "10 to 29", "30-59": "30 to 59", "60-119": "60 to 119", ">=120": r"$\geq$120"}


def read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cell(metric: str, value: str) -> str:
    if not value:
        return r"\tbd"
    if value == "n/a":
        return "n/a"
    v = float(value)
    if metric == "IDSW":
        return str(int(v))
    if metric == "CE":
        return "0.0" if v == 0 else f"{v:+.1f}"
    if metric == "RIP":
        return f"{v:.1f}"
    return f"{v:.2f}"


def best(metric: str, values: list[float]) -> float:
    if metric == "IDSW":
        return min(values)
    if metric == "CE":
        return min(values, key=abs)
    return max(values)


def bolded(metric: str, column: list[str]) -> list[str]:
    """Cells of one column within a group; bold the best only if every row has a number."""
    cells = [cell(metric, v) for v in column]
    values = [number(v) for v in column]
    if any(v is None for v in values):
        return cells
    top = best(metric, values)
    key = abs if metric == "CE" else (lambda x: x)
    return [rf"\textbf{{{c}}}" if key(v) == key(top) else c for c, v in zip(cells, values)]


def env(tex: str, label: str) -> tuple[int, int]:
    """Span of the table/figure environment holding \\label{label}."""
    at = tex.index(rf"\label{{{label}}}")
    start = max(tex.rfind(r"\begin{table}", 0, at), tex.rfind(r"\begin{figure}", 0, at))
    end = min(i for i in (tex.find(r"\end{table}", at), tex.find(r"\end{figure}", at)) if i != -1)
    return start, end


def fill_grouped(tex: str, label: str, rows: list[dict[str, str]], key: str, group_of) -> str:
    """tab:main / tab:seq: groups of four tracker rows, each `<lead> & <tracker> & cells \\\\`."""
    start, end = env(tex, label)
    block = tex[start:end]
    by_group: dict[str, dict[str, dict[str, str]]] = {}
    for r in rows:
        by_group.setdefault(group_of(r[key]), {})[r["tracker"]] = r
    lines, group = block.split("\n"), None
    for i, line in enumerate(lines):
        m = re.match(r"^(\\multirow\{4\}\{\*\}\{(.*)\}|\s*) & (" + "|".join(TRACKERS) + r") & .*\\\\\s*$", line)
        if not m:
            continue
        if m.group(2):
            group = m.group(2)
        tracker, data = m.group(3), by_group.get(group, {})
        if tracker not in data:
            continue
        columns = {mt: bolded(mt, [data.get(t, {}).get(mt, "") for t in TRACKERS]) for mt in METRICS}
        cells = [columns[mt][TRACKERS.index(tracker)] for mt in METRICS]
        lines[i] = f"{m.group(1)} & {tracker} & " + " & ".join(cells) + r" \\"
    return tex[:start] + "\n".join(lines) + tex[end:]


def fill_ablation(tex: str) -> str:
    rows = {r["variant"]: r for r in read("tab_ablation.csv")}
    columns = [c for c in next(iter(rows.values())) if c != "variant"]
    metric_of = {c: ("RIP" if c == "RIP" else c.rsplit("_", 1)[1]) for c in columns}
    cols = {c: bolded(metric_of[c], [rows[v].get(c, "") for v in rows]) for c in columns}
    variants = list(rows)
    start, end = env(tex, "tab:ablation")
    lines = tex[start:end].split("\n")
    for i, line in enumerate(lines):
        for v in variants:
            if line.startswith(v + " &"):
                lines[i] = f"{v} & " + " & ".join(cols[c][variants.index(v)] for c in columns) + r" \\"
    return tex[:start] + "\n".join(lines) + tex[end:]


def fill_runtime(tex: str) -> str:
    start, end = env(tex, "tab:runtime")
    lines = tex[start:end].split("\n")
    for r in read("tab_runtime.csv"):
        label = SEQ_LABELS.get(r["sequence"])
        for i, line in enumerate(lines):
            if label and line.startswith(label + " &"):
                fps = [f"{float(r[t]):.2f}" if r.get(t) else r"\tbd" for t in TRACKERS]
                lines[i] = f"{label} & {r['hardware']} & " + " & ".join(fps) + r" \\"
    return tex[:start] + "\n".join(lines) + tex[end:]


def fill_gate(tex: str) -> str:
    rows = read("fig_gate.csv")
    start, end = env(tex, "fig:gate")
    block = tex[start:end]
    panels = block.split(r"\end{tikzpicture}")
    for p, metric in enumerate(("HOTA", "RIP")):
        for seq, style in GATE_SERIES.items():
            pts = [(float(r["gate"]), r[metric]) for r in rows if r["sequence"] == seq and number(r[metric]) is not None]
            if not pts:
                continue
            coords = " ".join(f"({7.81 if abs(g - 7.8147) < 1e-3 else g:g},{float(v):.2f})" for g, v in sorted(pts))
            plot = rf"\addplot[{style}] coordinates {{{coords}}};"
            pattern = re.compile(r"\\addplot\[" + re.escape(style) + r"\] coordinates \{[^}]*\};")
            if pattern.search(panels[p]):
                panels[p] = pattern.sub(lambda _: plot, panels[p])
            elif p == 0 and rf"\addlegendimage{{{style}}}" in panels[p]:
                panels[p] = panels[p].replace(rf"\addlegendimage{{{style}}}", plot)
                panels[p] = panels[p].replace(r"Apple 0008 (\tbd)", "Apple 0008")
            elif p == 1:
                panels[p] = panels[p].replace(r"\end{axis}", plot + "\n" + r"\end{axis}", 1)
    return tex[:start] + r"\end{tikzpicture}".join(panels) + tex[end:]


def fill_gap(tex: str) -> str:
    rows = read("fig_gap.csv")
    order = ("StrongSORT", "BoT-SORT", "SAM2MOT", "MapSORT")
    present = [t for t in order if any(r["tracker"] == t for r in rows)]
    missing = [t for t in order if t not in present]
    if "MapSORT" not in present:
        print("fig:gap left as is: no MapSORT bins yet")
        return tex
    start, end = env(tex, "fig:gap")
    block = tex[start:end]
    # Plots sit between the axis options' closing "]" line and \end{axis}.
    opts_end = block.index("\n]\n", block.index(r"\begin{axis}[")) + len("\n]\n")
    head, rest = block[:opts_end], block[opts_end:]
    plots = []
    for t in present:
        coords = " ".join(
            f"({GAP_BINS[r['gap_bin']]},{float(r['RIP']):.1f})" for r in rows if r["tracker"] == t and number(r["RIP"]) is not None
        )
        plots.append(rf"\addplot[{GAP_STYLE[t]}] coordinates {{{coords}}};")
    legend = [rf"\addlegendentry{{{t}}}" for t in present]
    # Trackers without pooled bins yet keep a placeholder legend entry.
    legend += [rf"\addlegendimage{{area legend, {GAP_STYLE[t]}}}\addlegendentry{{{t} (\tbd)}}" for t in missing]
    body = "\n".join(plots) + "\n" + "\n".join(legend) + "\n"
    rest = rest[rest.index(r"\end{axis}"):]
    return tex[:start] + head + body + rest + tex[end:]


def main() -> None:
    tex = TEX.read_text()
    tex = fill_grouped(tex, "tab:main", read("tab_main.csv"), "dataset", lambda d: d)
    seq_group = {
        "LettuceMOTS 0011": r"\shortstack[l]{LettuceMOTS\\0011}",
        "MOrangeT V01": r"\shortstack[l]{MOrangeT\\V01}",
        "GrapeMOTS PathPlanning_7": r"\shortstack[l]{GrapeMOTS\\PathPlanning\_7}",
        "AppleMOTS 0008": r"\shortstack[l]{AppleMOTS\\0008}",
    }
    tex = fill_grouped(tex, "tab:seq", read("tab_sequences.csv"), "sequence", lambda s: seq_group[s])
    tex = fill_ablation(tex)
    tex = fill_runtime(tex)
    tex = fill_gate(tex)
    tex = fill_gap(tex)
    TEX.write_text(tex)
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
