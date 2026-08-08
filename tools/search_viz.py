#!/usr/bin/env python3
"""Visualize a luminal search run as a single self-contained HTML page.

Inputs (from one example run):
  --candidate-ops FILE   the LUMINAL_CANDIDATE_OPS log (one line per profiled
                         candidate: "cand=N <metric> [| filters] | op:count,...")
  --stdout-log FILE      the run's stdout with LUMINAL_LOG_LLIR=1 (LLIR_BEST
                         blocks: canonical node lines between LLIR_BEST ... and
                         LLIR_BEST_END)
  --out FILE             output HTML (default search_viz.html)

Produces a trajectory dashboard — metric vs candidate index per bucket, colored by
     kernel family, with the best-so-far front.

Usage:
  LUMINAL_LOG_LLIR=1 LUMINAL_CANDIDATE_OPS=ops.txt ./target/release/llama > run.log 2>&1
  python3 tools/search_viz.py --candidate-ops ops.txt --stdout-log run.log --out viz.html
"""

import argparse
import json
import re

METRIC_RE = re.compile(r"cand=(\d+)\s+(?:(\d+)\s*m?s\s*)?(?:(\d+)\s*ms\s*)?(?:(\d+)\s*µs)?")


def parse_metric_ms(tag: str) -> float | None:
    """Parse '<a> s <b> ms <c> µs' style durations after 'cand=N '."""
    m = re.search(r"cand=\d+\s+(.*)", tag)
    if not m:
        return None
    rest = m.group(1)
    total = 0.0
    seen = False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(µs|ms|s)\b", rest):
        seen = True
        v = float(value)
        total += v / 1000 if unit == "µs" else v * 1000 if unit == "s" else v
        # Only consume the leading duration tokens; stop at the first
        # non-duration content (e.g. memory / kernel list).
        if unit == "µs":
            break
    return total if seen else None



def parse_candidates(path: str):
    """Yield per-bucket candidate lists: [(index, metric_ms, ops_histogram)]."""
    buckets = []
    current = []
    last_cand = -1
    for line in open(path, errors="replace"):
        parts = line.rsplit("|", 1)
        if len(parts) != 2:
            continue
        tag, ops = parts[0].strip(), parts[1].strip()
        m = re.match(r"cand=(\d+)", tag)
        if not m:
            continue
        cand = int(m.group(1))
        if cand < last_cand and current:
            buckets.append(current)
            current = []
        last_cand = cand
        metric = parse_metric_ms(tag)
        if metric is None:
            continue
        current.append((cand, metric, ops))
    if current:
        buckets.append(current)
    return buckets


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>luminal search viz</title>
<style>
:root {{
  --bg: #0a0a0b; --panel: #111114; --border: #26262b;
  --text: #e8e8ea; --muted: #77777f; --accent: #00e589;
  --dot: #5b8dee; --mean: #ffb454; --front: #e8e8ea;
}}
* {{ box-sizing: border-box; }}
body {{ font: 13px/1.5 ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
       margin: 0; padding: 2.5em 3em; background: var(--bg); color: var(--text); }}
h1 {{ font-size: 20px; font-weight: 600; letter-spacing: 0.02em; margin: 0 0 0.2em; }}
h1::after {{ content: ""; display: block; width: 42px; height: 2px; background: var(--accent); margin-top: 10px; }}
h3 {{ font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.14em;
     color: var(--muted); margin: 2.2em 0 0.7em; }}
h3 b {{ color: var(--text); }}
.plot {{ background: var(--panel); border: 1px solid var(--border); margin-bottom: 0.4em; }}
.legend {{ margin: 1.6em 0 0.5em; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; }}
.legend span {{ margin-right: 1.6em; }}
.legend label {{ color: var(--muted); text-transform: none; letter-spacing: 0; }}
input[type=range] {{ accent-color: var(--accent); }}
#tip {{ position: fixed; display: none; background: #17171b; color: var(--text);
       border: 1px solid var(--border); padding: 6px 10px; font-size: 11px;
       pointer-events: none; z-index: 10; max-width: 340px; }}
.footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--border);
          color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; }}
.footer::before {{ content: "●"; color: var(--accent); margin-right: 0.6em; }}
</style></head><body>
<h1>SEARCH TRAJECTORY</h1>
<div id="tip"></div>
{trajectory}
<div class="footer">{footer}</div>
<script>
const data = {data_json};
const GEN_SIZE = {gen_size};
const DOT = "#5b8dee";
function drawAll(win) {{
  document.getElementById("winlabel").textContent = win;
  data.forEach((bucket, bi) => drawBucket(bucket, bi, win));
}}
function drawBucket(bucket, bi, WIN) {{
  const el = document.getElementById("plot" + bi);
  const W = el.clientWidth - 70, H = 260, PAD = 40;
  const xs = bucket.map(d => d[0]), ys = bucket.map(d => d[1]);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const logy = v => Math.log(Math.max(v, 1e-3));
  const y0 = logy(ymin), y1 = logy(ymax);
  const X = x => PAD + (x - xs[0]) / Math.max(1, xs[xs.length-1] - xs[0]) * W;
  const Y = v => 15 + (1 - (logy(v) - y0) / Math.max(1e-9, y1 - y0)) * (H - 30);
  let svg = `<svg width="${{W+70}}" height="${{H+30}}">`;
  // best-so-far front
  let best = Infinity, path = "";
  bucket.forEach(d => {{ if (d[1] < best) best = d[1]; path += (path?"L":"M") + X(d[0]) + "," + Y(best) + " "; }});
  svg += `<path d="${{path}}" fill="none" stroke="#e8e8ea" stroke-width="1.2" opacity="0.85"/>`;
  // Windowed running mean: how the population evolves, not just the
  // frontier. Window size comes from the slider.
  let mpath = "";
  for (let i = 0; i < bucket.length; i++) {{
    const lo = Math.max(0, i - WIN + 1);
    let sum = 0;
    for (let j = lo; j <= i; j++) sum += bucket[j][1];
    const mean = sum / (i - lo + 1);
    mpath += (mpath?"L":"M") + X(bucket[i][0]) + "," + Y(mean) + " ";
  }}
  svg += `<path d="${{mpath}}" fill="none" stroke="#ffb454" stroke-width="1.2" stroke-dasharray="5,3" opacity="0.9"/>`;
  let bestSeen = Infinity;
  bucket.forEach((d, di) => {{
    const isBest = d[1] < bestSeen;
    if (isBest) bestSeen = d[1];
    const fill = isBest ? "#00e589" : DOT;
    const r = isBest ? 5 : 3.5;
    const stroke = isBest ? ' stroke="#0a0a0b" stroke-width="1"' : "";
    svg += `<circle cx="${{X(d[0])}}" cy="${{Y(d[1])}}" r="${{r}}" fill="${{fill}}" opacity="0.9"${{stroke}}` +
           ` data-cand="${{d[0]}}" data-gen="${{Math.floor(di / GEN_SIZE)}}"` +
           ` data-ms="${{d[1].toFixed(3)}}" data-best="${{isBest ? "1" : ""}}" data-ops="${{d[2].replaceAll('"', "")}}"/>`;
  }});
  // Crosshair guides, hidden until hover.
  svg += `<line id="cross${{bi}}x" y1="15" y2="${{H-15}}" stroke="#aaa" stroke-width="1" opacity="0" pointer-events="none"/>` +
         `<line id="cross${{bi}}y" x1="${{PAD}}" x2="${{PAD+W}}" stroke="#aaa" stroke-width="1" opacity="0" pointer-events="none"/>`;
  [ymin, ymax].forEach(v => {{
    svg += `<text x="2" y="${{Y(v)+4}}" font-size="10" fill="#77777f">${{v.toFixed(2)}}ms</text>`;
  }});
  svg += `<text x="${{PAD}}" y="${{H+25}}" font-size="10" fill="#77777f">candidate 0 .. ${{xs[xs.length-1]}} (log-y)</text></svg>`;
  el.innerHTML = svg;
}}
drawAll(25);
document.getElementById("winslider").addEventListener("input", e => drawAll(+e.target.value));
const tip = document.getElementById("tip");
document.addEventListener("mouseover", e => {{
  if (e.target.tagName !== "circle" || !e.target.dataset.cand) return;
  const d = e.target.dataset;
  tip.innerHTML = `candidate ${{d.cand}} &middot; generation ${{d.gen}}` +
                  `${{d.best ? " &middot; new best" : ""}}<br>${{d.ms}} ms<br>` +
                  `<span style="opacity:0.7">${{d.ops.split(",").slice(0, 8).join(", ")}}${{d.ops.split(",").length > 8 ? ", …" : ""}}</span>`;
  tip.style.display = "block";
  const plot = e.target.closest(".plot");
  const bi = plot.id.replace("plot", "");
  const vx = document.getElementById("cross" + bi + "x");
  const vy = document.getElementById("cross" + bi + "y");
  vx.setAttribute("x1", e.target.getAttribute("cx"));
  vx.setAttribute("x2", e.target.getAttribute("cx"));
  vy.setAttribute("y1", e.target.getAttribute("cy"));
  vy.setAttribute("y2", e.target.getAttribute("cy"));
  vx.setAttribute("opacity", "0.25");
  vy.setAttribute("opacity", "0.25");
}});
document.addEventListener("mousemove", e => {{
  if (tip.style.display === "block") {{
    tip.style.left = (e.clientX + 14) + "px";
    tip.style.top = (e.clientY - 10) + "px";
  }}
}});
document.addEventListener("mouseout", e => {{
  if (e.target.tagName !== "circle") return;
  tip.style.display = "none";
  document.querySelectorAll("[id^=cross]").forEach(l => l.setAttribute("opacity", "0"));
}});
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-ops", required=True)
    ap.add_argument("--out", default="search_viz.html")
    ap.add_argument(
        "--stdout-log",
        help="run stdout; used to pull bucket dimension labels ('Group i/n: ...' lines)",
    )
    ap.add_argument(
        "--generation-size", type=int, default=10,
        help="offspring per generation (CompileOptions::generation_size) — used to label generations",
    )
    args = ap.parse_args()

    labels = []
    if args.stdout_log:
        for raw in open(args.stdout_log, errors="replace"):
            for line in raw.replace("\r", "\n").split("\n"):
                m = re.search(r"Group \d+/\d+: (.+)$", line)
                if m:
                    labels.append(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", m.group(1)).strip())

    buckets = parse_candidates(args.candidate_ops)
    data = [[(c, m, ops) for c, m, ops in b] for b in buckets]

    legend = (
        '<div class="legend">'
        '<span style="color:var(--accent)">&#9679; new best</span>'
        '<span style="color:var(--dot)">&#9679; candidate</span>'
        '<span style="color:var(--front)">&#9473; best-so-far</span>'
        '<span style="color:var(--mean)">&#9476; running mean</span>'
        ' <label>window: <input id="winslider" type="range" min="2" max="100" value="25"'
        ' style="vertical-align:middle"> <b id="winlabel">25</b></label></div>'
    )
    trajectory = legend
    for i, b in enumerate(buckets):
        dims = f" &middot; {labels[i]}" if i < len(labels) else ""
        trajectory += (
            f"<h3>{i:02d} / <b>bucket {i}</b>{dims} &middot; {len(b)} candidates</h3>"
            f"<div class='plot' id='plot{i}'></div>"
        )

    open(args.out, "w").write(
        PAGE.format(
            trajectory=trajectory,
            data_json=json.dumps(data),
            gen_size=args.generation_size,
            footer=f"{sum(len(b) for b in buckets)} candidates &middot; {len(buckets)} buckets &middot; generation size {args.generation_size}",
        )
    )
    print(f"wrote {args.out}: {len(buckets)} buckets, {sum(len(b) for b in buckets)} candidates")


if __name__ == "__main__":
    main()
