#!/usr/bin/env python3
"""Render Luminal invocation JSONL as a self-contained HTML report.

The trace is produced by setting LUMINAL_PROFILE_JSONL while running any
CompiledModel workload. This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path

OUTER_FIELDS = [
    ("setup", "CompiledModel setup", "host"),
    ("dynamic_dims", "Dynamic dimensions", "host"),
    ("input_bind", "Input binding", "host"),
    ("stream_handoff", "Stream handoff", "host"),
    ("output_metadata", "Output metadata", "host"),
    ("output_plan", "Output planning", "host"),
]
RUNTIME_FIELDS = [
    ("dispatch", "Runtime dispatch", "host"),
    ("prepare", "Runtime prepare (inclusive)", "inclusive"),
    ("resource_signature", "Resource signature", "host"),
    ("resource_validation", "Resource validation", "host"),
    ("arena_allocate", "Arena allocation", "host"),
    ("refresh_lengths", "Refresh lengths", "host"),
    ("collect_hlir", "Collect dirty HLIR", "host"),
    ("resolve_input_ptrs", "Resolve input pointers", "host"),
    ("install_input_ptrs", "Install input pointers", "host"),
    ("output_registration", "Output registration", "host"),
    ("materialize", "Graph materialization/update", "host"),
    ("buffer_map", "HostOp buffer maps", "host"),
    ("graph_launch", "CUDA graph launch API", "host"),
    ("hostop_launch", "Ordinary HostOp API", "host"),
    ("output_copy", "Output copies", "host"),
    ("sync", "Stream execution/wait", "wait"),
    ("stats", "Runtime stats", "host"),
    ("cleanup", "Runtime cleanup", "host"),
]
TAIL_FIELDS = [
    ("runtime_boundary", "PyO3/runtime boundary (nested)", "inclusive"),
    ("output_finalize", "Output finalization", "host"),
    ("unattributed", "CompiledModel unattributed", "unknown"),
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"profile trace does not exist: {path}\n"
            "Run a Luminal workload with "
            f"LUMINAL_PROFILE_JSONL={path} before rendering it."
        )
    records = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if record.get("kind") == "luminal_invocation":
                records.append(record)
    records.sort(key=lambda record: record["invocation"])
    if not records:
        raise SystemExit(f"no luminal_invocation records found in {path}")
    return records


def value(record: dict, scope: str, field: str) -> float:
    if scope == "runtime":
        runtime = record.get("runtime") or {}
        return float(runtime.get("timings_us", {}).get(field, 0.0))
    return float(record.get("compiled_model", {}).get("timings_us", {}).get(field, 0.0))


def representative(records: list[dict], phase: str) -> dict | None:
    candidates = [record for record in records if record.get("phase") == phase]
    # The first call for a graph may contain cold materialization. Prefer warm
    # calls when the trace contains enough observations.
    warm = [record for record in candidates if record.get("call_index", 0) > 0]
    candidates = warm or candidates
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: abs(
            value(record, "compiled_model", "total")
            - statistics.median(
                value(candidate, "compiled_model", "total") for candidate in candidates
            )
        ),
    )


def color(elapsed: float, maximum: float, category: str) -> str:
    strength = 0.0 if maximum <= 0 else math.sqrt(max(0.0, elapsed) / maximum)
    lightness = 98 - 50 * strength
    hue = {"host": 210, "wait": 28, "inclusive": 275, "unknown": 0}[category]
    return f"hsl({hue} 78% {lightness:.1f}%)"


def render(records: list[dict], source: Path) -> str:
    rows = [("compiled_model", *row) for row in OUTER_FIELDS]
    rows += [("runtime", *row) for row in RUNTIME_FIELDS]
    rows += [("compiled_model", *row) for row in TAIL_FIELDS]
    maximum = max(
        value(record, scope, field) for scope, field, _, _ in rows for record in records
    )

    summaries = []
    for scope, field, label, category in rows:
        values = [value(record, scope, field) for record in records]
        summaries.append(
            (
                label,
                category,
                len(values),
                statistics.mean(values),
                percentile(values, 0.5),
                percentile(values, 0.9),
                percentile(values, 0.99),
                statistics.pstdev(values) if len(values) > 1 else 0.0,
            )
        )

    heat_header = "".join(
        f"<th title='invocation {r['invocation']}, graph {r['graph']}, call {r['call_index']}'>"
        f"{html.escape(r.get('phase', '?')[:1].upper())}{r['call_index']}</th>"
        for r in records
    )
    heat_rows = []
    for scope, field, label, category in rows:
        cells = "".join(
            f"<td style='background:{color(value(r, scope, field), maximum, category)}' "
            f"title='{html.escape(label)}: {value(r, scope, field):,.1f} us'>"
            f"{value(r, scope, field):,.0f}</td>"
            for r in records
        )
        heat_rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")

    summary_rows = "".join(
        "<tr>"
        + "".join(
            [
                f"<th>{html.escape(label)}</th>",
                f"<td>{category}</td>",
                f"<td>{count}</td>",
            ]
            + [f"<td>{number:,.1f}</td>" for number in (mean, p50, p90, p99, stdev)]
        )
        + "</tr>"
        for label, category, count, mean, p50, p90, p99, stdev in summaries
    )

    waterfall_parts = []
    # These are mutually exclusive at the outer level except that graph_run is
    # replaced by the runtime's broad top-level regions.
    waterfall_fields = [
        ("compiled_model", "setup", "setup", "host"),
        ("compiled_model", "dynamic_dims", "dims", "host"),
        ("compiled_model", "input_bind", "bind", "host"),
        ("compiled_model", "output_metadata", "metadata", "host"),
        ("compiled_model", "output_plan", "output plan", "host"),
        ("compiled_model", "runtime_boundary", "runtime boundary", "host"),
        ("runtime", "dispatch", "dispatch", "host"),
        ("runtime", "prepare", "prepare", "host"),
        ("runtime", "output_registration", "out register", "host"),
        ("runtime", "materialize", "materialize", "host"),
        ("runtime", "buffer_map", "buffer map", "host"),
        ("runtime", "graph_launch", "graph API", "host"),
        ("runtime", "hostop_launch", "HostOp API", "host"),
        ("runtime", "output_copy", "copies", "host"),
        ("runtime", "sync", "stream wait", "wait"),
        ("runtime", "stats", "stats", "host"),
        ("runtime", "cleanup", "cleanup", "host"),
        ("compiled_model", "output_finalize", "finalize", "host"),
        ("compiled_model", "unattributed", "unattributed", "unknown"),
    ]
    for phase in ("prefill", "decode"):
        record = representative(records, phase)
        if record is None:
            continue
        parts = [
            (label, value(record, scope, field), category)
            for scope, field, label, category in waterfall_fields
        ]
        total = sum(part[1] for part in parts) or 1.0
        bars = "".join(
            f"<span class='bar {category}' style='width:{elapsed / total * 100:.4f}%' "
            f"title='{html.escape(label)}: {elapsed:,.1f} us'></span>"
            for label, elapsed, category in parts
            if elapsed > 0
        )
        legend = ", ".join(
            f"{html.escape(label)} {elapsed:,.1f} us"
            for label, elapsed, _ in parts
            if elapsed >= total * 0.01
        )
        waterfall_parts.append(
            f"<h3>{phase.title()} invocation {record['invocation']} (graph {record['graph']}, call {record['call_index']})</h3>"
            f"<div class='waterfall'>{bars}</div><p class='small'>{legend}</p>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Luminal invocation profile</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#172033}} h1,h2{{margin-top:28px}}
table{{border-collapse:collapse}} th,td{{border:1px solid #d8deea;padding:5px 8px;text-align:right;white-space:nowrap}}
th:first-child{{text-align:left;position:sticky;left:0;background:white;z-index:1}} .scroll{{overflow:auto;max-width:100%}}
.waterfall{{display:flex;width:100%;height:34px;background:#eee;border-radius:4px;overflow:hidden}} .bar{{min-width:1px}}
.host{{background:#3b82f6}} .wait{{background:#f59e0b}} .unknown{{background:#ef4444}} .inclusive{{background:#a855f7}}
.legend span{{display:inline-block;padding:4px 8px;margin-right:8px;color:white;border-radius:3px}} .small{{font-size:12px;color:#536079}}
</style></head><body>
<h1>Luminal invocation profile</h1>
<p>Source: <code>{html.escape(str(source))}</code>. Records: {len(records)}. All values are microseconds.</p>
<p class="legend"><span class="host">host-active/API</span><span class="wait">stream execution/wait</span><span class="inclusive">inclusive diagnostic</span><span class="unknown">unattributed</span></p>
<p><strong>Important:</strong> runtime <em>prepare</em> is inclusive of its detailed rows; CompiledModel <em>graph_run</em> is inclusive of the entire runtime and is omitted from the heatmap to avoid visual double counting. Stream wait includes GPU execution and dependency waiting; it is not removable CPU bookkeeping.</p>
<h2>Representative invocation waterfall</h2>{"".join(waterfall_parts) or "<p>No confidently classified prefill/decode records.</p>"}
<h2>Invocation heatmap</h2><div class="scroll"><table><thead><tr><th>Region</th>{heat_header}</tr></thead><tbody>{"".join(heat_rows)}</tbody></table></div>
<h2>Summary</h2><table><thead><tr><th>Region</th><th>class</th><th>n</th><th>mean</th><th>p50</th><th>p90</th><th>p99</th><th>stddev</th></tr></thead><tbody>{summary_rows}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="LUMINAL_PROFILE_JSONL output")
    parser.add_argument("-o", "--output", type=Path, help="HTML destination")
    args = parser.parse_args()
    output = args.output or args.trace.with_suffix(".html")
    records = load_records(args.trace)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(records, args.trace))
    print(f"wrote {output} ({len(records)} invocations)")


if __name__ == "__main__":
    main()
