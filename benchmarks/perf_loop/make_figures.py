# SPDX-License-Identifier: MIT

"""Draw the figures of ``docs/perf_loop.md`` from the recorded measurements.

    python benchmarks/perf_loop/make_figures.py

Reads ``accepted_changes.json`` and ``end_to_end.json`` next to this file and
writes two SVG files to ``docs/img/``.  Standard library only, so the figures
can be regenerated anywhere; the measurements themselves were taken once, on
the machine named in the JSON files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs" / "img"

INK, MUTED, GRID, LINE, DOT, PANEL = "#1f2933", "#52606d", "#d9e2ec", "#3b6fd4", "#1f9d55", "#ffffff"
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return "\n".join([
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' role='img' aria-label='{title}'>",
        f"<title>{title}</title>",
        f"<rect width='{width}' height='{height}' rx='10' fill='{PANEL}' stroke='{GRID}'/>",
        *body, "</svg>", ""])


def _text(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}' {FONT}>{s}</text>")


def cumulative(data: dict) -> str:
    """Step chart: cumulative speedup of the kept changes against the iteration."""
    kept = [c for c in data["changes"] if c["kept"]]
    n_iter = data["iterations"]
    W, H, L, R, T, B = 860, 400, 70, 28, 54, 54
    top = math.ceil((kept[-1]["cumulative"] - 1) * 100 / 25) * 25
    x = lambda it: L + (W - L - R) * it / n_iter
    y = lambda pct: H - B - (H - T - B) * pct / top
    body = [_text(L, 30, "Cumulative speedup of the accepted changes", 16, weight="600")]
    for pct in range(0, top + 1, 25):
        body.append(f"<line x1='{L}' x2='{W - R}' y1='{y(pct):.1f}' y2='{y(pct):.1f}' stroke='{GRID}'/>")
        body.append(_text(L - 10, y(pct) + 4, f"+{pct}%", 12, MUTED, "end"))
    # the objective grew twice: shade and label the three periods
    spans = []
    for c in kept:
        if not spans or spans[-1][0] != c["workloads"]:
            spans.append([c["workloads"], c["iteration"]])
    starts = [0] + [s[1] - 1 for s in spans[1:]] + [n_iter]
    for i, (workloads, _) in enumerate(spans):
        x0, x1 = x(starts[i]), x(starts[i + 1])
        if i % 2:
            body.append(f"<rect x='{x0:.1f}' y='{T}' width='{x1 - x0:.1f}' height='{H - T - B}' "
                        f"fill='{GRID}' opacity='0.35'/>")
        if x1 - x0 > 60:
            body.append(_text((x0 + x1) / 2, H - B + 34, f"{workloads} workloads", 12, MUTED, "middle"))
    pts, level = [(x(0), y(0))], 0.0
    for c in kept:
        pts.append((x(c["iteration"]), y(level)))
        level = (c["cumulative"] - 1) * 100
        pts.append((x(c["iteration"]), y(level)))
    pts.append((x(n_iter), y(level)))
    body.append("<polyline fill='none' stroke='%s' stroke-width='2.2' points='%s'/>"
                % (LINE, " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)))
    for c in kept:
        body.append(f"<circle cx='{x(c['iteration']):.1f}' cy='{y((c['cumulative'] - 1) * 100):.1f}' "
                    f"r='3.6' fill='{DOT}'/>")
    # name the four largest steps: to the left of the step, or below its new
    # level when there is no room on the left
    for c in sorted(kept, key=lambda c: -c["speedup"])[:4]:
        px, py = x(c["iteration"]), y((c["cumulative"] - 1) * 100)
        label = f"{c['change']} (&#215;{c['speedup']:.2f})"
        if px - 9 - 6.6 * (len(label) - 6) < L:
            body.append(_text(px + 9, py + 20, label, 12, INK))
        else:
            body.append(_text(px - 9, py + 4, label, 12, INK, "end"))
    for it in (0, 100, 200, 300, n_iter):
        body.append(_text(x(it), H - B + 16, str(it), 12, MUTED, "middle"))
    body.append(_text(W - R, H - 8, "loop iteration", 12, MUTED, "end"))
    return _svg(W, H, body, "Cumulative speedup of the accepted changes by loop iteration")


def end_to_end(data: dict) -> str:
    """Horizontal bars on a log axis: speedup per workload, before against after."""
    rows = data["by_workload"]
    W, L, R, T, row_h = 860, 230, 80, 62, 44
    H = T + row_h * len(rows) + 50
    top = max(r["speedup"] for r in rows)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if t <= top * 1.6]
    span = math.log(ticks[-1])
    x = lambda v: L + (W - L - R) * math.log(max(v, 1.0)) / span
    body = [_text(22, 30, "Release 3.4.1 against the optimized code, by workload", 16, weight="600"),
            _text(22, 48, "geometric mean of time before / time after over the scenarios that have the "
                          "workload (log scale)", 12, MUTED)]
    for t in ticks:
        body.append(f"<line x1='{x(t):.1f}' x2='{x(t):.1f}' y1='{T}' y2='{H - 44}' stroke='{GRID}'/>")
        body.append(_text(x(t), H - 26, f"&#215;{t}", 12, MUTED, "middle"))
    for i, r in enumerate(rows):
        cy = T + row_h * i + row_h / 2
        body.append(_text(L - 12, cy - 2, r["label"], 13, INK, "end", "600"))
        body.append(_text(L - 12, cy + 14, r["what"], 11, MUTED, "end"))
        body.append(f"<rect x='{L}' y='{cy - 11:.1f}' width='{max(x(r['speedup']) - L, 2):.1f}' height='22' "
                    f"rx='3' fill='{LINE}'/>")
        body.append(_text(x(r["speedup"]) + 8, cy + 5, f"&#215;{r['speedup']:.2f}", 13, INK, weight="600"))
    return _svg(W, H, body, "Speedup of the optimized code over release 3.4.1 by workload")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    changes = json.loads((HERE / "accepted_changes.json").read_text())
    (OUT / "perf_loop_cumulative.svg").write_text(cumulative(changes))
    e2e = json.loads((HERE / "end_to_end.json").read_text())
    (OUT / "perf_loop_end_to_end.svg").write_text(end_to_end(e2e))
    print("wrote", OUT / "perf_loop_cumulative.svg", "and", OUT / "perf_loop_end_to_end.svg")


if __name__ == "__main__":
    main()
