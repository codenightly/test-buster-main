"""Tests for console rendering and every report writer."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.console import Console

from testbuster import html_report
from testbuster.config import REPORTED_PERCENTILES, Gates, RunPlan
from testbuster.errors import TestBusterError
from testbuster.metrics import Attempt, Report, Tally, pct_key
from testbuster.reporting import (
    CSV_COLUMNS,
    PHASE_SERIES,
    NdjsonWriter,
    human_bytes,
    human_duration,
    human_ms,
    print_banner,
    render_header,
    render_prometheus,
    render_report,
    report_to_json,
    status_style,
    terminal_handles_blocks,
    write_csv,
    write_json,
    write_prometheus,
)


def _report(
    make_attempt: Callable[..., Attempt],
    *,
    count: int = 10,
    keep: bool = False,
    gates: Gates | None = None,
    dns: float | None = 0.002,
    connect: float | None = 0.004,
) -> Report:
    tally = Tally(keep_attempts=keep)
    for index in range(count):
        tally.record(
            make_attempt(status=200, elapsed=0.010 + index * 0.001, dns=dns, connect=connect)
        )
    plan = RunPlan(
        target="https://example.com",
        total_requests=count,
        keep_attempts=keep,
        gates=gates or Gates(),
    )
    return tally.summarize(plan, 1.5, interrupted=False)


class TestHumanBytes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.00 KiB"),
            (1536, "1.50 KiB"),
            (1024 * 1024, "1.00 MiB"),
            (1024 * 1024 * 1024, "1.00 GiB"),
        ],
    )
    def test_picks_a_readable_unit(self, value: float, expected: str) -> None:
        assert human_bytes(value) == expected


class TestHumanMs:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, "500 us"), (12.5, "12.50 ms"), (1500.0, "1.500 s")],
    )
    def test_scales_the_unit(self, value: float, expected: str) -> None:
        assert human_ms(value) == expected


class TestHumanDuration:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.25, "250 ms"), (5.5, "5.500 s"), (90.0, "1m 30.0s")],
    )
    def test_scales_the_unit(self, value: float, expected: str) -> None:
        assert human_duration(value) == expected


class TestStatusStyle:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0, "bright_red"),
            (200, "green"),
            (301, "cyan"),
            (404, "yellow"),
            (500, "red"),
        ],
    )
    def test_maps_a_code_to_a_color(self, code: int, expected: str) -> None:
        assert status_style(code) == expected


class TestRenderReport:
    @staticmethod
    def capture(report: Report) -> str:
        console = Console(record=True, width=100, no_color=True)
        render_report(console, report)
        return console.export_text()

    def test_shows_the_headline_numbers(self, make_attempt: Callable[..., Attempt]) -> None:
        text = self.capture(_report(make_attempt, count=10))
        assert "summary" in text
        assert "latency (total)" in text
        assert "status codes" in text
        assert "200" in text

    def test_lists_percentiles_in_a_fixed_order(self, make_attempt: Callable[..., Attempt]) -> None:
        # The order is fixed, so two reports diff cleanly.
        text = self.capture(_report(make_attempt, count=50))
        positions = [text.index(key) for key in ("p50", "p75", "p90", "p95", "p99")]
        assert positions == sorted(positions)

    def test_repeated_renders_match(self, make_attempt: Callable[..., Attempt]) -> None:
        report = _report(make_attempt, count=20)
        assert self.capture(report) == self.capture(report)

    def test_shows_phase_tables_when_samples_exist(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        text = self.capture(_report(make_attempt, count=5))
        assert "dns lookup" in text
        assert "connect and tls" in text

    def test_hides_phase_tables_without_samples(self, make_attempt: Callable[..., Attempt]) -> None:
        text = self.capture(_report(make_attempt, count=5, dns=None, connect=None))
        assert "dns lookup" not in text

    def test_reports_an_empty_run_plainly(self) -> None:
        empty = Tally().summarize(RunPlan(target="https://example.com"), 1.0, interrupted=False)
        assert "no requests completed" in self.capture(empty)

    def test_flags_an_interrupted_run(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt())
        report = tally.summarize(RunPlan(target="https://example.com"), 1.0, interrupted=True)
        assert "stopped early" in self.capture(report)

    def test_shows_a_failure_table_only_when_something_failed(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        clean = self.capture(_report(make_attempt, count=5))
        assert "failures" not in clean

        tally = Tally()
        tally.record(make_attempt(status=0, failure="timeout"))
        broken = tally.summarize(RunPlan(target="https://example.com"), 1.0, interrupted=False)
        assert "timeout" in self.capture(broken)

    def test_shows_gate_results(self, make_attempt: Callable[..., Attempt]) -> None:
        text = self.capture(_report(make_attempt, count=10, gates=Gates(max_p95_ms=1)))
        assert "gates" in text
        assert "FAIL" in text


class TestRenderHeader:
    def test_lists_the_plan(self) -> None:
        console = Console(record=True, width=100, no_color=True)
        plan = RunPlan(target="https://example.com", workers=25, total_requests=500)
        render_header(console, plan, "asyncio")
        text = console.export_text()

        assert "https://example.com" in text
        assert "25" in text
        assert "500 requests" in text
        assert "asyncio" in text

    def test_prints_advisories(self) -> None:
        console = Console(record=True, width=100, no_color=True)
        plan = RunPlan(target="https://example.com", verify_tls=False)
        render_header(console, plan, "asyncio")
        assert "TLS verification is off" in console.export_text()


class TestJsonOutput:
    def test_produces_parseable_json(self, make_attempt: Callable[..., Attempt]) -> None:
        payload = json.loads(report_to_json(_report(make_attempt)))
        assert payload["schema"] == "testbuster/report/1"
        assert payload["summary"]["total_requests"] == 10

    def test_compact_mode_is_one_line(self, make_attempt: Callable[..., Attempt]) -> None:
        assert "\n" not in report_to_json(_report(make_attempt), indent=None)

    def test_writes_a_file(self, make_attempt: Callable[..., Attempt], tmp_path: Path) -> None:
        destination = tmp_path / "out" / "report.json"
        write_json(_report(make_attempt), destination)

        assert destination.exists()
        saved = json.loads(destination.read_text(encoding="utf-8"))
        assert saved["summary"]["total_requests"] == 10

    def test_keeps_secrets_out_of_the_file(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        tally = Tally()
        tally.record(make_attempt())
        plan = RunPlan(target="https://example.com", headers={"Authorization": "Bearer s3cret"})
        report = tally.summarize(plan, 1.0, interrupted=False)

        destination = tmp_path / "report.json"
        write_json(report, destination)
        assert "s3cret" not in destination.read_text(encoding="utf-8")


class TestCsvOutput:
    def test_writes_one_row_per_attempt(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        destination = tmp_path / "attempts.csv"
        write_csv(_report(make_attempt, count=7, keep=True), destination)

        with destination.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 7
        assert list(rows[0]) == list(CSV_COLUMNS)
        assert rows[0]["index"] == "1"
        assert rows[-1]["index"] == "7"
        assert rows[0]["status"] == "200"

    def test_explains_the_missing_records(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        with pytest.raises(TestBusterError, match="--save-attempts"):
            write_csv(_report(make_attempt, keep=False), tmp_path / "attempts.csv")


def _timeline_report(make_attempt: Callable[..., Attempt], *, count: int = 20) -> Report:
    tally = Tally()
    for index in range(count):
        tally.record(make_attempt(status=200, elapsed=0.01 + index * 0.001), second=index // 5)
    return tally.summarize(RunPlan(target="https://example.com"), 2.0, interrupted=False)


class TestPrometheus:
    def test_has_help_and_type_lines(self, make_attempt: Callable[..., Attempt]) -> None:
        text = render_prometheus(_timeline_report(make_attempt))
        assert "# HELP testbuster_requests_total" in text
        assert "# TYPE testbuster_requests_total counter" in text
        assert 'testbuster_requests_total{target="https://example.com"} 20' in text

    def test_carries_latency_quantiles(self, make_attempt: Callable[..., Attempt]) -> None:
        text = render_prometheus(_timeline_report(make_attempt))
        assert 'quantile="95"' in text

    def test_escapes_label_values(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt())
        report = tally.summarize(RunPlan(target='https://x/"quote"'), 1.0, interrupted=False)
        text = render_prometheus(report)
        assert '\\"quote\\"' in text

    def test_writes_a_file(self, make_attempt: Callable[..., Attempt], tmp_path: Path) -> None:
        dest = tmp_path / "m.prom"
        write_prometheus(_timeline_report(make_attempt), dest)
        assert "testbuster_requests_total" in dest.read_text(encoding="utf-8")


class TestHtml:
    def test_is_a_full_document(self, make_attempt: Callable[..., Attempt]) -> None:
        html = html_report.render(_timeline_report(make_attempt), tool_version="9.9.9")
        assert html.startswith("<!doctype html>")
        assert "Test Buster! report" in html
        assert "9.9.9" in html

    def test_has_a_histogram_and_timeline_svg(self, make_attempt: Callable[..., Attempt]) -> None:
        html = html_report.render(_timeline_report(make_attempt), tool_version="1.0.0")
        assert html.count("<svg") >= 2  # histogram and timeline

    def test_escapes_the_target(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt())
        report = tally.summarize(RunPlan(target="https://x/<script>"), 1.0, interrupted=False)
        html = html_report.render(report, tool_version="1.0.0")
        assert "<script>" not in html.split("<style>")[0] + html.split("</style>")[-1]
        assert "&lt;script&gt;" in html

    def test_writes_a_file(self, make_attempt: Callable[..., Attempt], tmp_path: Path) -> None:
        dest = tmp_path / "out" / "r.html"
        html_report.write(_timeline_report(make_attempt), dest, tool_version="1.0.0")
        assert dest.exists()
        assert dest.read_text(encoding="utf-8").startswith("<!doctype html>")


class TestNdjson:
    def test_writes_one_line_per_attempt(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        dest = tmp_path / "a.ndjson"
        with NdjsonWriter(dest) as writer:
            for _ in range(3):
                writer.write(make_attempt(status=200))

        lines = dest.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["status"] == 200

    def test_dash_writes_to_stdout(self, make_attempt: Callable[..., Attempt], monkeypatch) -> None:
        buffer = io.StringIO()
        monkeypatch.setattr("sys.stdout", buffer)
        writer = NdjsonWriter("-")
        writer.write(make_attempt(status=201))
        writer.close()  # must not close stdout
        assert json.loads(buffer.getvalue().strip())["status"] == 201
        assert not buffer.closed


def _stalled_report(make_attempt: Callable[..., Attempt]) -> Report:
    """A run with traffic in second 0 and second 3, and a stall in between."""
    tally = Tally()
    for second in (0, 0, 3, 3):
        tally.record(make_attempt(status=200), second=second)
    return tally.summarize(RunPlan(target="https://example.com"), 4.0, interrupted=False)


def _polyline_ys(document: str) -> list[float]:
    """Pull the y value of every plotted point out of the throughput chart."""
    match = re.search(r'<polyline points="([^"]+)"', document)
    assert match is not None
    return [float(pair.split(",")[1]) for pair in match.group(1).split(" ")]


class TestPhaseSeriesReachEveryRenderer:
    """Every timing series the report carries must reach every renderer.

    The dns series shipped missing from the HTML report and from the Prometheus
    text, because each renderer listed the series by hand.
    """

    def test_prometheus_carries_the_dns_series(self, make_attempt: Callable[..., Attempt]) -> None:
        assert "testbuster_dns_ms" in render_prometheus(_report(make_attempt, count=5))

    def test_html_carries_a_dns_row(self, make_attempt: Callable[..., Attempt]) -> None:
        document = html_report.render(_report(make_attempt, count=5), tool_version="1.0.0")
        assert "<td>dns</td>" in document

    def test_both_renderers_list_every_series(self, make_attempt: Callable[..., Attempt]) -> None:
        report = _report(make_attempt, count=5)
        text = render_prometheus(report)
        document = html_report.render(report, tool_version="1.0.0")
        for name in PHASE_SERIES:
            assert f"testbuster_{name}_ms" in text
            assert f"<td>{name}</td>" in document

    def test_an_empty_series_stays_out(self, make_attempt: Callable[..., Attempt]) -> None:
        report = _report(make_attempt, count=5, dns=None)
        assert "testbuster_dns_ms" not in render_prometheus(report)
        assert "<td>dns</td>" not in html_report.render(report, tool_version="1.0.0")


class TestHtmlPercentileColumns:
    """The columns come from REPORTED_PERCENTILES, so a change cannot drop one."""

    def test_every_reported_percentile_gets_a_column(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        document = html_report.render(_report(make_attempt, count=5), tool_version="1.0.0")
        for pct in REPORTED_PERCENTILES:
            assert f"<th>{pct_key(pct)}</th>" in document


class TestHtmlTileColor:
    """The pass or fail color belongs on the outcome tiles, not on the grid."""

    def test_the_grid_carries_no_verdict(self, make_attempt: Callable[..., Attempt]) -> None:
        document = html_report.render(_report(make_attempt, count=5), tool_version="1.0.0")
        assert '<div class="tiles">' in document

    def test_a_clean_run_marks_the_outcome_tiles(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        document = html_report.render(_report(make_attempt, count=5), tool_version="1.0.0")
        assert '<div class="k">success rate</div><div class="v ok">' in document
        assert '<div class="k">failed</div><div class="v ok">' in document

    def test_a_failed_run_tints_only_the_outcome_tiles(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        tally = Tally()
        tally.record(make_attempt(status=500))
        report = tally.summarize(RunPlan(target="https://example.com"), 1.0, interrupted=False)
        document = html_report.render(report, tool_version="1.0.0")

        assert '<div class="k">success rate</div><div class="v bad">' in document
        assert '<div class="k">failed</div><div class="v bad">' in document
        # A byte count and a wall time report no verdict, so they stay plain.
        assert '<div class="k">data in</div><div class="v">' in document
        assert '<div class="k">wall time</div><div class="v">' in document
        assert document.count('class="v bad"') == 2


class TestThroughputChart:
    """An idle second must plot as zero, or a stall reads as sustained traffic."""

    def test_idle_seconds_reach_the_baseline(self, make_attempt: Callable[..., Attempt]) -> None:
        document = html_report.render(_stalled_report(make_attempt), tool_version="1.0.0")
        ys = _polyline_ys(document)

        assert len(ys) == 4  # seconds 0 to 3, with the two idle ones filled in
        top, baseline = min(ys), max(ys)
        assert ys == [top, baseline, baseline, top]

    def test_a_busy_run_still_plots_every_second(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        ys = _polyline_ys(html_report.render(_timeline_report(make_attempt), tool_version="1.0.0"))
        assert len(ys) == 4  # 20 requests over seconds 0 to 3


class TestBanner:
    def test_wide_utf8_console_draws_block_art(self) -> None:
        # A real UTF-8 stream lets the block glyphs through, so the art renders.
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        console = Console(file=stream, width=120, force_terminal=True, color_system=None)
        print_banner(console)
        stream.flush()
        text = stream.buffer.getvalue().decode("utf-8")  # type: ignore[attr-defined]
        assert "█" in text

    def test_narrow_console_uses_one_line(self) -> None:
        console = Console(record=True, width=40)
        print_banner(console)
        text = console.export_text()
        assert "Test Buster!" in text
        assert "█" not in text

    def test_a_plain_code_page_falls_back(self) -> None:
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        console = Console(file=stream, width=120, force_terminal=True, color_system=None)
        print_banner(console)
        stream.flush()
        text = stream.buffer.getvalue().decode("cp1252")  # type: ignore[attr-defined]
        assert "Test Buster!" in text
        assert "█" not in text

    def test_encoding_probe(self) -> None:
        assert terminal_handles_blocks(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")) is True
        assert terminal_handles_blocks(io.TextIOWrapper(io.BytesIO(), encoding="cp1252")) is False
