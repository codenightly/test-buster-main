"""Rendering: console tables, the JSON report, and the CSV of attempts.

Every table sorts its rows before printing, so percentile and status-code
lines come out in the same order on each run and two reports diff cleanly.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final

from rich.box import SIMPLE_HEAD
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from testbuster import APP_NAME, __version__
from testbuster.config import REPORTED_PERCENTILES, RunPlan
from testbuster.errors import TestBusterError
from testbuster.metrics import Attempt, Report, Spread, pct_key

_KIB: Final[float] = 1024.0
_MIB: Final[float] = 1024.0 * 1024.0
_GIB: Final[float] = 1024.0 * 1024.0 * 1024.0


def human_bytes(count: float) -> str:
    """Format a byte count with a unit a person can read at a glance."""
    if count < _KIB:
        return f"{count:.0f} B"
    if count < _MIB:
        return f"{count / _KIB:.2f} KiB"
    if count < _GIB:
        return f"{count / _MIB:.2f} MiB"
    return f"{count / _GIB:.2f} GiB"


def human_ms(milliseconds: float) -> str:
    """Format a millisecond value, dropping to microseconds when it is tiny."""
    if milliseconds < 1.0:
        return f"{milliseconds * 1000:.0f} us"
    if milliseconds < 1000.0:
        return f"{milliseconds:.2f} ms"
    return f"{milliseconds / 1000:.3f} s"


def human_duration(seconds: float) -> str:
    """Format a wall-clock span."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}m {rest:.1f}s"


def status_style(code: int) -> str:
    """Pick a color for a status code so a wall of 500s stands out."""
    if code == 0:
        return "bright_red"
    if 200 <= code < 300:
        return "green"
    if 300 <= code < 400:
        return "cyan"
    if 400 <= code < 500:
        return "yellow"
    return "red"


def render_header(console: Console, plan: RunPlan, backend: str) -> None:
    """Print what the run is about to do, before the first request goes out."""
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="dim", justify="right")
    table.add_column(style="bold")

    table.add_row("target", plan.target)
    table.add_row("method", plan.method)
    table.add_row("concurrency", str(plan.workers))
    table.add_row("stop after", plan.stop_label)
    table.add_row("timeout", human_duration(plan.timeout))
    if plan.profile is not None:
        table.add_row("profile", plan.profile.describe())
    elif plan.rate_limit is not None:
        table.add_row("rate cap", f"{plan.rate_limit:g} req/s")
    if plan.warmup:
        table.add_row("warmup", f"{plan.warmup} requests (not measured)")
    if plan.retries:
        table.add_row(
            "retries", f"{plan.retries} per request, {plan.retry_backoff:g}s base backoff"
        )
    if plan.proxy:
        table.add_row("proxy", plan.proxy)
    if plan.http2:
        table.add_row("transport", "HTTP/2 (httpx)")
    if plan.cookies:
        table.add_row("cookies", "kept across the run")
    if plan.compact_memory:
        table.add_row("memory", "compact (histogram percentiles)")
    if not plan.expectations.is_empty:
        table.add_row("checks", "response validation on")
    table.add_row("event loop", backend)

    console.print(table)

    for note in plan.advisories():
        console.print(f"[yellow]note[/] {note}")
    console.print()


#: Every timing series a report carries, in the order readers expect. One list
#: keeps the console tables, the HTML rows, and the Prometheus series in step.
PHASE_SERIES: Final[tuple[str, ...]] = ("latency", "ttfb", "dns", "connect")

#: Console table title for each name in PHASE_SERIES.
_PHASE_TITLES: Final[dict[str, str]] = {
    "latency": "latency (total)",
    "ttfb": "time to first byte",
    "dns": "dns lookup",
    "connect": "connect and tls",
}


def phase_spreads(report: Report) -> list[tuple[str, Spread]]:
    """Pair every timing series with its name, in PHASE_SERIES order.

    A renderer that iterates this cannot forget a series the report carries.
    """
    pairs: list[tuple[str, Spread]] = []
    for name in PHASE_SERIES:
        spread: Spread = getattr(report, name)
        pairs.append((name, spread))
    return pairs


def _spread_table(title: str, spread: Spread) -> Table:
    """Build one timing table. Percentile rows follow REPORTED_PERCENTILES."""
    table = Table(title=title, box=SIMPLE_HEAD, title_justify="left", title_style="bold")
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right")

    if spread.count == 0:
        table.add_row("samples", "0")
        return table

    table.add_row("min", human_ms(spread.min_ms))
    table.add_row("mean", human_ms(spread.mean_ms))
    table.add_row("stdev", human_ms(spread.stdev_ms))
    for pct in REPORTED_PERCENTILES:
        key = pct_key(pct)
        table.add_row(key, human_ms(spread.percentiles_ms[key]))
    table.add_row("max", human_ms(spread.max_ms))
    return table


def _summary_table(report: Report) -> Table:
    table = Table(box=SIMPLE_HEAD, title="summary", title_justify="left", title_style="bold")
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right")

    ok_style = "green" if report.failed == 0 else "yellow"
    table.add_row("requests", f"{report.total:,}")
    table.add_row(
        "successful",
        Text(f"{report.succeeded:,}  ({report.success_rate:.2f}%)", style=ok_style),
    )
    table.add_row(
        "failed",
        Text(
            f"{report.failed:,}  ({report.failure_rate:.2f}%)",
            style="red" if report.failed else "dim",
        ),
    )
    if report.retried:
        table.add_row("needed a retry", f"{report.retried:,}")
    table.add_row("wall time", human_duration(report.wall_seconds))
    table.add_row("throughput", f"{report.requests_per_second:,.2f} req/s")
    table.add_row("data in", human_bytes(report.bytes_in))
    table.add_row("per request", human_bytes(report.bytes_per_request))
    table.add_row("bandwidth", f"{human_bytes(report.throughput_bytes_per_second)}/s")
    return table


def _status_table(report: Report) -> Table:
    table = Table(box=SIMPLE_HEAD, title="status codes", title_justify="left", title_style="bold")
    table.add_column("code", style="dim")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")

    for code, count in report.status_counts.items():
        share = count / report.total * 100 if report.total else 0.0
        label = "no response" if code == 0 else str(code)
        table.add_row(Text(label, style=status_style(code)), f"{count:,}", f"{share:.1f}%")
    return table


def _failure_table(report: Report) -> Table:
    table = Table(box=SIMPLE_HEAD, title="failures", title_justify="left", title_style="bold")
    table.add_column("reason", style="red")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")

    for reason, count in report.failure_counts.items():
        share = count / report.total * 100 if report.total else 0.0
        table.add_row(reason, f"{count:,}", f"{share:.1f}%")
    return table


@dataclass(frozen=True, slots=True)
class ChartBars:
    """Bucketed latency counts and the labels a chart draws around them."""

    bars: list[tuple[float, float, int]]
    peak: int
    low_ms: float
    high_ms: float


def chart_bars(report: Report, buckets: int) -> ChartBars | None:
    """Bucket the latency histogram for a chart, or None with no samples.

    The console sparkline and the HTML bar chart scale against the same peak
    and print the same end labels, so one function derives them.
    """
    bars = report.histogram.bars(buckets=buckets)
    if not bars:
        return None
    return ChartBars(
        bars=bars,
        peak=max(count for _, _, count in bars) or 1,
        low_ms=bars[0][0] * 1000,
        high_ms=bars[-1][1] * 1000,
    )


_SPARK = " ▁▂▃▄▅▆▇█"


def _sparkline(report: Report, width: int = 40) -> Text:
    """Draw the latency histogram as one line of block characters."""
    chart = chart_bars(report, width)
    if chart is None:
        return Text("")
    glyphs = []
    for _low, _high, count in chart.bars:
        level = 0 if count == 0 else 1 + int((count / chart.peak) * (len(_SPARK) - 2))
        glyphs.append(_SPARK[min(level, len(_SPARK) - 1)])
    line = Text()
    line.append(f"{human_ms(chart.low_ms):>10}  ", style="dim")
    line.append("".join(glyphs), style="cyan")
    line.append(f"  {human_ms(chart.high_ms)}", style="dim")
    return line


def _label_table(report: Report) -> Table:
    table = Table(box=SIMPLE_HEAD, title="by step", title_justify="left", title_style="bold")
    table.add_column("step", style="dim")
    table.add_column("requests", justify="right")
    table.add_column("success", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")

    for name, stats in report.by_label.items():
        data = stats.to_dict()
        rate_style = "green" if data["failed"] == 0 else "yellow"
        table.add_row(
            name,
            f"{data['total']:,}",
            Text(f"{data['success_rate_pct']:.1f}%", style=rate_style),
            human_ms(data["p50_ms"]),
            human_ms(data["p95_ms"]),
            human_ms(data["p99_ms"]),
        )
    return table


def _gate_table(report: Report) -> Table:
    table = Table(box=SIMPLE_HEAD, title="gates", title_justify="left", title_style="bold")
    table.add_column("gate", style="dim")
    table.add_column("limit", justify="right")
    table.add_column("actual", justify="right")
    table.add_column("result", justify="right")

    for gate in report.check_gates():
        comparison = "max" if gate.unit == "ms" else "min"
        table.add_row(
            gate.name,
            f"{comparison} {gate.limit:g} {gate.unit}",
            f"{gate.actual:.2f} {gate.unit}",
            Text("pass", style="green") if gate.passed else Text("FAIL", style="bold red"),
        )
    return table


def render_report(console: Console, report: Report) -> None:
    """Print the finished report."""
    if report.total == 0:
        console.print(
            "[yellow]no requests completed.[/] Nothing to measure. "
            "Check the target URL and the network path."
        )
        return

    if report.stop_reason == "max errors reached":
        console.print(
            "[bold red]run stopped:[/] the failure cap was reached. The numbers "
            "below cover only the requests that finished.\n"
        )
    elif report.interrupted:
        console.print(
            "[yellow]run stopped early.[/] The numbers below cover only the "
            "requests that finished.\n"
        )

    blocks: list[Any] = [
        _summary_table(report),
        _spread_table(_PHASE_TITLES["latency"], report.latency),
    ]

    # The sparkline uses block glyphs. Skip it when the console encoding cannot
    # carry them, so an old Windows code page never crashes the report.
    if terminal_handles_blocks(console.file):
        spark = _sparkline(report)
        if spark.plain:
            blocks.append(Text("latency shape", style="bold"))
            blocks.append(spark)

    for name, spread in phase_spreads(report):
        # Total latency already has a table above the shape line. A phase with
        # no samples means the run never measured it, so it gets no table.
        if name == "latency" or spread.count == 0:
            continue
        blocks.append(_spread_table(_PHASE_TITLES[name], spread))

    if report.by_label:
        blocks.append(_label_table(report))

    blocks.append(_status_table(report))
    if report.failure_counts:
        blocks.append(_failure_table(report))
    if report.plan.gates.any_set:
        blocks.append(_gate_table(report))

    console.print(
        Panel(
            Group(*blocks),
            title="[bold]Test Buster! results[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def report_to_json(report: Report, *, indent: int | None = 2) -> str:
    """Serialize the report. Pass indent=None for one compact line."""
    return json.dumps(report.to_dict(tool_version=__version__), indent=indent, sort_keys=False)


def write_text(destination: Path, text: str) -> None:
    """Save text as UTF-8, creating parent directories as needed.

    A failed write reports one plain line instead of a traceback. The report
    writers share this path, so each one fails the same way.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise TestBusterError(f"cannot write {destination}: {exc.strerror}") from exc


def write_json(report: Report, destination: Path) -> None:
    """Save the JSON report."""
    write_text(destination, report_to_json(report) + "\n")


#: Column order for the CSV of attempts. Keep it stable for downstream scripts.
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "index",
    "label",
    "status",
    "elapsed_ms",
    "ttfb_ms",
    "dns_ms",
    "connect_ms",
    "bytes_in",
    "retries",
    "failure",
    "validation_error",
)


def write_csv(report: Report, destination: Path) -> None:
    """Save one row per recorded attempt.

    This needs --save-attempts, which the CLI turns on whenever --csv is used.
    """
    if not report.attempts:
        raise TestBusterError(
            "no per-request records were kept. Add --save-attempts to write a CSV"
        )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for index, attempt in enumerate(report.attempts, start=1):
                row = attempt.to_dict()
                row["index"] = index
                writer.writerow(
                    {
                        column: "" if row.get(column) is None else row[column]
                        for column in CSV_COLUMNS
                    }
                )
    except OSError as exc:
        raise TestBusterError(f"cannot write {destination}: {exc.strerror}") from exc


class NdjsonWriter:
    """Writes one JSON line per attempt as the run proceeds.

    A long run can stream every request to a file or to stdout, so a pipeline
    consumes results without waiting for the whole run to finish. Pass "-" for
    stdout. The engine calls write() from its single event loop, so no lock is
    needed.
    """

    def __init__(self, destination: Path | str) -> None:
        self._own = destination != "-"
        self._handle: IO[str]
        if not self._own:
            self._handle = sys.stdout
        else:
            path = Path(destination)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = path.open("w", encoding="utf-8")
            except OSError as exc:
                raise TestBusterError(f"cannot write {path}: {exc.strerror}") from exc

    def write(self, attempt: Attempt) -> None:
        self._handle.write(json.dumps(attempt.to_dict(), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._own:
            self._handle.close()

    def __enter__(self) -> NdjsonWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ----------------------------------------------------------------- the banner

# "TEST BUSTER!" as block letters. Each string is one row of the whole
# wordmark, not one letter, so the art stays in one place.
_BIG: Final[tuple[str, ...]] = (
    "████████╗███████╗███████╗████████╗    ██████╗ ██╗   ██╗███████╗████████╗███████╗██████╗ ██╗",
    "╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██║",
    "   ██║   █████╗  ███████╗   ██║       ██████╔╝██║   ██║███████╗   ██║   █████╗  ██████╔╝██║",
    "   ██║   ██╔══╝  ╚════██║   ██║       ██╔══██╗██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗╚═╝",
    "   ██║   ███████╗███████║   ██║       ██████╔╝╚██████╔╝███████║   ██║   ███████╗██║  ██║██╗",
    "   ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝",
)

# One color per row, cool at the crown and warm at the base.
_ROW_COLORS: Final[tuple[str, ...]] = (
    "bright_cyan",
    "cyan",
    "blue",
    "bright_magenta",
    "magenta",
    "bright_red",
)

#: The block art is this many columns wide. Below it, the tool prints one line.
BIG_BANNER_WIDTH: Final[int] = 92

_TAGLINE: Final[str] = "async HTTP load generation with phase timings and CI gates"


def terminal_handles_blocks(stream: IO[str] | None) -> bool:
    """Report whether the stream encoding can carry the block characters."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "█╔╗╚╝║".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _one_line(console: Console) -> None:
    """Print the compact banner for a narrow terminal or a plain code page."""
    console.print(f"[bold cyan]{APP_NAME}[/] [dim]v{__version__}[/]  {_TAGLINE}")


def print_banner(console: Console) -> None:
    """Draw the block art, or one text line when the terminal cannot show it."""
    if console.width < BIG_BANNER_WIDTH or not terminal_handles_blocks(console.file):
        _one_line(console)
        return

    console.print()
    for index, line in enumerate(_BIG):
        console.print(Text(line, style=_ROW_COLORS[index % len(_ROW_COLORS)]))
    console.print(f"[bold yellow]{_TAGLINE}[/]  [dim]v{__version__}[/]")
    console.print()


# ------------------------------------------------------- prometheus exposition

_PREFIX = "testbuster"


def _escape(value: str) -> str:
    """Escape a Prometheus label value."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_prometheus(report: Report) -> str:
    """Return the Prometheus text exposition for a report."""
    target = _escape(report.plan.target)
    label = f'{{target="{target}"}}'
    lines: list[str] = []

    def metric(name: str, help_text: str, value: float, kind: str = "gauge") -> None:
        full = f"{_PREFIX}_{name}"
        lines.append(f"# HELP {full} {help_text}")
        lines.append(f"# TYPE {full} {kind}")
        lines.append(f"{full}{label} {value}")

    metric("requests_total", "Requests sent.", report.total, kind="counter")
    metric("requests_succeeded_total", "Requests that succeeded.", report.succeeded, kind="counter")
    metric("requests_failed_total", "Requests that failed.", report.failed, kind="counter")
    metric(
        "requests_retried_total", "Requests that needed a retry.", report.retried, kind="counter"
    )
    metric("success_rate_percent", "Share of requests that succeeded.", report.success_rate)
    metric("throughput_rps", "Requests per second over the run.", report.requests_per_second)
    metric("bytes_in_total", "Response bytes read.", report.bytes_in, kind="counter")
    metric("wall_seconds", "Wall-clock length of the run.", report.wall_seconds)

    for name, spread in phase_spreads(report):
        if spread.count == 0:
            continue
        full = f"{_PREFIX}_{name}_ms"
        lines.append(f"# HELP {full} {name} timing in milliseconds.")
        lines.append(f"# TYPE {full} gauge")
        for pct_name, value in spread.percentiles_ms.items():
            quantile = pct_name[1:].replace("_", ".")
            lines.append(f'{full}{{target="{target}",quantile="{quantile}"}} {value}')

    for code, count in report.status_counts.items():
        full = f"{_PREFIX}_status_total"
        lines.append(f'{full}{{target="{target}",code="{code}"}} {count}')

    return "\n".join(lines) + "\n"


def write_prometheus(report: Report, destination: Path) -> None:
    """Write the Prometheus exposition to a file."""
    write_text(destination, render_prometheus(report))
