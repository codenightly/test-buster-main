"""A self-contained HTML report.

A p99 number hides the shape of a distribution. A bimodal latency, where half
the requests are fast and half are slow, shows the same p99 as a smooth one. A
chart makes the difference plain.

This writes one HTML file with no external assets: inline CSS, and charts drawn
as inline SVG. It opens in any browser and survives being emailed or attached to
a build. The page follows the reader's light or dark preference.
"""

from __future__ import annotations

import html
from pathlib import Path

from testbuster.config import REPORTED_PERCENTILES
from testbuster.metrics import Report, pct_key
from testbuster.reporting import (
    chart_bars,
    human_bytes,
    human_duration,
    human_ms,
    phase_spreads,
    write_text,
)

_CSS = """
:root {
  --bg: #ffffff; --fg: #1c1f24; --muted: #667085; --card: #f5f6f8; --line: #e2e5ea;
  --accent: #2f6feb; --ok: #1f9d55; --warn: #c9820a; --bad: #d64545;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e9ef; --muted: #98a2b3; --card: #171a21;
    --line: #262b34; --accent: #6ea8fe; --ok: #4ade80; --warn: #fbbf24; --bad: #f87171;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1000px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .2rem; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .8rem; }
.sub { color: var(--muted); margin: 0 0 1.6rem; word-break: break-all; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .9rem 1rem; }
.tile .k { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.tile .v { font-size: 1.35rem; font-weight: 650; margin-top: .25rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .4rem .7rem; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: .8rem; }
.scroll { overflow-x: auto; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); }
svg { max-width: 100%; height: auto; display: block; }
.bar { fill: var(--accent); }
.axis { stroke: var(--line); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 11px; }
.foot { color: var(--muted); font-size: .8rem; margin-top: 3rem; }
"""


def _tile(key: str, value: str, *, value_class: str = "") -> str:
    """Render one tile. value_class colors this number and nothing else."""
    number = f"v {value_class}" if value_class else "v"
    return f'<div class="tile"><div class="k">{html.escape(key)}</div><div class="{number}">{html.escape(value)}</div></div>'


def _histogram_svg(report: Report, width: int = 920, height: int = 220) -> str:
    """Draw the latency histogram as an SVG bar chart."""
    chart = chart_bars(report, 48)
    if chart is None:
        return "<p class='sub'>No latency samples.</p>"

    bars = chart.bars
    pad_l, pad_b, pad_t = 44, 26, 10
    plot_w = width - pad_l - 8
    plot_h = height - pad_b - pad_t
    slot = plot_w / len(bars)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="latency histogram">']
    parts.append(
        f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - 8}" y2="{pad_t + plot_h}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}"/>'
    )

    for index, (low, _high, count) in enumerate(bars):
        bar_h = (count / chart.peak) * plot_h
        x = pad_l + index * slot
        y = pad_t + plot_h - bar_h
        title = f"{human_ms(low * 1000)}: {count}"
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{max(slot - 1, 1):.1f}" '
            f'height="{bar_h:.1f}"><title>{html.escape(title)}</title></rect>'
        )

    low_label = html.escape(human_ms(chart.low_ms))
    high_label = html.escape(human_ms(chart.high_ms))
    parts.append(f'<text class="tick" x="{pad_l}" y="{height - 8}">{low_label}</text>')
    parts.append(
        f'<text class="tick" x="{width - 8}" y="{height - 8}" text-anchor="end">{high_label}</text>'
    )
    parts.append(f'<text class="tick" x="4" y="{pad_t + 10}">{chart.peak}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _timeline_svg(report: Report, width: int = 920, height: int = 180) -> str:
    """Draw requests-per-second over the run as an SVG area chart."""
    if not report.timeline:
        return ""

    # The timeline holds only the seconds that saw traffic. Plot it as it comes
    # and a line jumps the gap, so a stall reads as steady load. Fill the idle
    # seconds with zero first, and a stall shows as the hole it was.
    recorded = dict(report.timeline)
    span = max(recorded) + 1
    counts = [recorded.get(second, 0) for second in range(span)]
    peak = max(counts) or 1
    pad_l, pad_b, pad_t = 44, 26, 10
    plot_w = width - pad_l - 8
    plot_h = height - pad_b - pad_t

    def point(second: int, count: int) -> tuple[float, float]:
        x = pad_l + (second / max(span - 1, 1)) * plot_w
        y = pad_t + plot_h - (count / peak) * plot_h
        return x, y

    dots = [point(second, count) for second, count in enumerate(counts)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in dots)
    area = f"{pad_l},{pad_t + plot_h} " + line + f" {dots[-1][0]:.1f},{pad_t + plot_h}"

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="throughput over time">']
    parts.append(f'<polygon points="{area}" fill="var(--accent)" opacity="0.18"/>')
    parts.append(f'<polyline points="{line}" fill="none" stroke="var(--accent)" stroke-width="2"/>')
    parts.append(
        f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - 8}" y2="{pad_t + plot_h}"/>'
    )
    parts.append(f'<text class="tick" x="{pad_l}" y="{height - 8}">0s</text>')
    parts.append(
        f'<text class="tick" x="{width - 8}" y="{height - 8}" text-anchor="end">{span}s</text>'
    )
    parts.append(f'<text class="tick" x="4" y="{pad_t + 10}">{peak}/s</text>')
    parts.append("</svg>")
    return "".join(parts)


def _spread_rows(report: Report) -> str:
    """Build the timing table: one row per series, one column per percentile."""
    order = [pct_key(pct) for pct in REPORTED_PERCENTILES]
    head = "<tr><th>series</th><th>min</th><th>mean</th>"
    head += "".join(f"<th>{p}</th>" for p in order) + "<th>max</th></tr>"
    rows = [head]
    for name, spread in phase_spreads(report):
        if spread.count == 0:
            continue
        cells = [f"<td>{html.escape(name)}</td>", f"<td>{human_ms(spread.min_ms)}</td>"]
        cells.append(f"<td>{human_ms(spread.mean_ms)}</td>")
        cells += [f"<td>{human_ms(spread.percentiles_ms[p])}</td>" for p in order]
        cells.append(f"<td>{human_ms(spread.max_ms)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(rows)


def _status_rows(report: Report) -> str:
    rows = ["<tr><th>code</th><th>count</th><th>share</th></tr>"]
    for code, count in report.status_counts.items():
        share = count / report.total * 100 if report.total else 0.0
        label = "no response" if code == 0 else str(code)
        klass = "ok" if 200 <= code < 300 else ("warn" if 300 <= code < 500 else "bad")
        rows.append(
            f'<tr><td class="{klass}">{label}</td><td>{count:,}</td><td>{share:.1f}%</td></tr>'
        )
    return "".join(rows)


def _label_rows(report: Report) -> str:
    if not report.by_label:
        return ""
    rows = [
        "<tr><th>step</th><th>requests</th><th>success</th><th>p50</th><th>p95</th><th>p99</th></tr>"
    ]
    for name, stats in report.by_label.items():
        data = stats.to_dict()
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{data['total']:,}</td>"
            f"<td>{data['success_rate_pct']:.1f}%</td>"
            f"<td>{human_ms(data['p50_ms'])}</td><td>{human_ms(data['p95_ms'])}</td>"
            f"<td>{human_ms(data['p99_ms'])}</td></tr>"
        )
    return "".join(rows)


def render(report: Report, *, tool_version: str) -> str:
    """Return the full HTML document for a report."""
    plan = report.plan
    # Only the two tiles that report outcome carry the pass or fail color. A
    # tinted grid would say a byte count or a wall time also passed or failed.
    ok_class = "ok" if report.failed == 0 else ("warn" if report.success_rate >= 95 else "bad")

    tiles = "".join(
        [
            _tile("requests", f"{report.total:,}"),
            _tile("success rate", f"{report.success_rate:.2f}%", value_class=ok_class),
            _tile("throughput", f"{report.requests_per_second:,.1f} req/s"),
            _tile("p95 latency", human_ms(report.latency.percentiles_ms["p95"])),
            _tile("p99 latency", human_ms(report.latency.percentiles_ms["p99"])),
            _tile("wall time", human_duration(report.wall_seconds)),
            _tile("data in", human_bytes(report.bytes_in)),
            _tile("failed", f"{report.failed:,}", value_class=ok_class),
        ]
    )

    timeline = _timeline_svg(report)
    timeline_block = f"<h2>Throughput over time</h2>{timeline}" if timeline else ""

    failures_block = ""
    if report.failure_counts:
        rows = ["<tr><th>reason</th><th>count</th></tr>"]
        for reason, count in report.failure_counts.items():
            rows.append(f"<tr><td>{html.escape(reason)}</td><td>{count:,}</td></tr>")
        failures_block = (
            f'<h2>Failures</h2><div class="scroll"><table>{"".join(rows)}</table></div>'
        )

    labels_block = ""
    label_rows = _label_rows(report)
    if label_rows:
        labels_block = f'<h2>By step</h2><div class="scroll"><table>{label_rows}</table></div>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test Buster! report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>Test Buster! report</h1>
  <p class="sub">{html.escape(plan.method)} {html.escape(plan.target)} &middot; v{html.escape(tool_version)}</p>

  <div class="tiles">{tiles}</div>

  <h2>Latency distribution</h2>
  {_histogram_svg(report)}

  <h2>Timing percentiles</h2>
  <div class="scroll"><table>{_spread_rows(report)}</table></div>

  {timeline_block}

  <h2>Status codes</h2>
  <div class="scroll"><table>{_status_rows(report)}</table></div>

  {failures_block}
  {labels_block}

  <p class="foot">Generated by Test Buster! v{html.escape(tool_version)}.</p>
</div>
</body>
</html>
"""


def write(report: Report, destination: Path, *, tool_version: str) -> None:
    """Write the HTML report to a file."""
    write_text(destination, render(report, tool_version=tool_version))
