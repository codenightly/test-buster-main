"""CLI tests: argv handling, subcommands, exit codes, and output writers."""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from testbuster import APP_NAME, __version__
from testbuster.cli import _emit, app, main, merge_headers
from testbuster.cluster import NO_WORKERS
from testbuster.config import Gates, RunPlan
from testbuster.errors import ExitCode, TestBusterError
from testbuster.metrics import Attempt, Report, Tally

runner = CliRunner()


def _printed(result: Any) -> str:
    """Return every line a command printed, whichever stream took it.

    One runner mixes stderr into output. Another keeps the two apart.
    """
    seen = [result.output]
    with contextlib.suppress(ValueError):
        seen.append(result.stderr)
    return "".join(seen)


def _report(
    make_attempt: Callable[..., Attempt],
    *,
    count: int = 10,
    oks: int | None = None,
    keep: bool = False,
    gates: Gates | None = None,
    interrupted: bool = False,
) -> Report:
    successes = count if oks is None else oks
    tally = Tally(keep_attempts=keep)
    for index in range(count):
        tally.record(make_attempt(status=200 if index < successes else 500))
    plan = RunPlan(
        target="https://example.com",
        total_requests=count,
        keep_attempts=keep,
        gates=gates or Gates(),
    )
    return tally.summarize(plan, 1.0, interrupted=interrupted)


class TestMergeHeaders:
    def test_reads_the_curl_form(self) -> None:
        assert merge_headers(["Accept: text/plain"]) == {"Accept": "text/plain"}

    def test_reads_the_json_form(self) -> None:
        assert merge_headers(['{"Accept": "text/plain"}']) == {"Accept": "text/plain"}

    def test_mixes_both_forms(self) -> None:
        merged = merge_headers(['{"A": "1"}', "B: 2"])
        assert merged == {"A": "1", "B": "2"}

    def test_a_later_flag_wins(self) -> None:
        assert merge_headers(["A: 1", "A: 2"])["A"] == "2"

    def test_no_flags_gives_no_headers(self) -> None:
        assert merge_headers([]) == {}


class TestArgvRewrite:
    """A positional URL and subcommands cannot share one group, so main()
    inserts the implicit `run`. These pin that behavior down."""

    @staticmethod
    def captured(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr("testbuster.cli.app", lambda args: seen.extend(args))
        main(argv)
        return seen

    def test_inserts_run_before_a_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.captured(monkeypatch, ["https://example.com"]) == [
            "run",
            "https://example.com",
        ]

    def test_inserts_run_before_a_leading_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.captured(monkeypatch, ["-n", "5", "https://x"]) == [
            "run",
            "-n",
            "5",
            "https://x",
        ]

    def test_no_args_shows_the_group_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A bare invocation lists every subcommand, not just run's help.
        assert self.captured(monkeypatch, []) == ["--help"]

    def test_a_help_flag_stays_at_the_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.captured(monkeypatch, ["--help"]) == ["--help"]

    @pytest.mark.parametrize("name", ["version", "completion", "run", "diff", "cluster"])
    def test_keeps_a_subcommand_intact(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        assert self.captured(monkeypatch, [name]) == [name]

    def test_scenario_becomes_a_run_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A scenario is a run whose requests come from a file, so the shorter
        # spelling rewrites onto `run --scenario` rather than a second command.
        assert self.captured(monkeypatch, ["scenario", "flow.json", "-n", "5"]) == [
            "run",
            "--scenario",
            "flow.json",
            "-n",
            "5",
        ]

    def test_a_bare_scenario_keeps_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No file: typer reports that --scenario needs an argument.
        assert self.captured(monkeypatch, ["scenario"]) == ["run", "--scenario"]

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_scenario_help_asks_the_run_command(
        self, monkeypatch: pytest.MonkeyPatch, flag: str
    ) -> None:
        # Passed on, the flag would name a file called "--help".
        assert self.captured(monkeypatch, ["scenario", flag]) == ["run", "--help"]

    def test_a_help_flag_after_the_file_stays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.captured(monkeypatch, ["scenario", "flow.json", "--help"]) == [
            "run",
            "--scenario",
            "flow.json",
            "--help",
        ]


class TestVersionCommand:
    def test_prints_the_name_and_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert APP_NAME in result.stdout
        assert __version__ in result.stdout

    def test_plain_mode_prints_only_the_number(self) -> None:
        result = runner.invoke(app, ["version", "--plain"])
        assert result.exit_code == 0
        assert result.stdout.strip() == __version__

    def test_does_not_run_a_load_test(self) -> None:
        # The old layout let the positional URL eat this word and load test
        # https://version. Guard against that coming back.
        result = runner.invoke(app, ["version"])
        assert "status codes" not in result.stdout


class TestCompletionCommand:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish", "powershell"])
    def test_emits_a_script(self, shell: str) -> None:
        result = runner.invoke(app, ["completion", shell])
        assert result.exit_code == 0
        assert len(result.stdout.strip()) > 0
        assert "testbuster" in result.stdout.lower()

    def test_rejects_an_unknown_shell(self) -> None:
        result = runner.invoke(app, ["completion", "tcsh"])
        assert result.exit_code != 0


class TestUsageErrors:
    def test_no_url_exits_two(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_a_bad_method_exits_two(self) -> None:
        result = runner.invoke(app, ["run", "https://example.com", "-X", "PSOT"])
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_a_bad_duration_exits_two(self) -> None:
        result = runner.invoke(app, ["run", "https://example.com", "-D", "soon"])
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_a_bad_url_scheme_exits_two(self) -> None:
        result = runner.invoke(app, ["run", "ftp://example.com"])
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_zero_concurrency_exits_two(self) -> None:
        result = runner.invoke(app, ["run", "https://example.com", "-c", "0"])
        assert result.exit_code == ExitCode.BAD_USAGE


class TestEmitExitCodes:
    @staticmethod
    def emit(report: Report, **kwargs: object) -> ExitCode:
        out = Console(record=True, width=100, no_color=True)
        err = Console(record=True, width=100, no_color=True, stderr=True)
        defaults: dict[str, object] = {
            "as_json": False,
            "json_path": None,
            "csv_path": None,
            "html_path": None,
            "prometheus_path": None,
        }
        defaults.update(kwargs)
        return _emit(report, out=out, err=err, **defaults)  # type: ignore[arg-type]

    def test_a_clean_run_exits_zero(self, make_attempt: Callable[..., Attempt]) -> None:
        assert self.emit(_report(make_attempt)) == ExitCode.OK

    def test_an_empty_run_exits_three(self) -> None:
        empty = Tally().summarize(RunPlan(target="https://example.com"), 1.0, interrupted=False)
        assert self.emit(empty) == ExitCode.RUN_FAILED

    def test_a_total_wipeout_exits_three(self, make_attempt: Callable[..., Attempt]) -> None:
        # Every request failed. Reporting success here would hide a dead target.
        assert self.emit(_report(make_attempt, count=10, oks=0)) == ExitCode.RUN_FAILED

    def test_one_success_is_enough_to_leave_three(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        assert self.emit(_report(make_attempt, count=10, oks=1)) == ExitCode.OK

    def test_a_failed_gate_exits_four(self, make_attempt: Callable[..., Attempt]) -> None:
        report = _report(make_attempt, count=10, gates=Gates(max_p95_ms=0.001))
        assert self.emit(report) == ExitCode.GATE_FAILED

    def test_a_passed_gate_exits_zero(self, make_attempt: Callable[..., Attempt]) -> None:
        report = _report(make_attempt, count=10, gates=Gates(max_p95_ms=10_000))
        assert self.emit(report) == ExitCode.OK

    def test_a_success_gate_beats_the_wipeout_rule_order(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        # Zero successes is reported as RUN_FAILED, not GATE_FAILED, because the
        # run produced nothing worth measuring against a threshold.
        report = _report(make_attempt, count=10, oks=0, gates=Gates(min_success_rate=99))
        assert self.emit(report) == ExitCode.RUN_FAILED

    def test_an_interrupted_run_exits_one_thirty(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        report = _report(make_attempt, count=5, interrupted=True)
        assert self.emit(report) == ExitCode.INTERRUPTED


class TestEmitOutputs:
    @staticmethod
    def emit(report: Report, **kwargs: object) -> ExitCode:
        return TestEmitExitCodes.emit(report, **kwargs)

    def test_writes_the_json_file(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        destination = tmp_path / "report.json"
        self.emit(_report(make_attempt), json_path=destination)

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["summary"]["total_requests"] == 10

    def test_writes_the_csv_file(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        destination = tmp_path / "attempts.csv"
        self.emit(_report(make_attempt, keep=True), csv_path=destination)

        lines = destination.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 11  # one header plus ten rows

    def test_a_csv_without_records_explains_itself(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        with pytest.raises(TestBusterError, match="--save-attempts"):
            self.emit(_report(make_attempt, keep=False), csv_path=tmp_path / "a.csv")

    def test_writes_both_files_at_once(
        self, make_attempt: Callable[..., Attempt], tmp_path: Path
    ) -> None:
        self.emit(
            _report(make_attempt, keep=True),
            json_path=tmp_path / "r.json",
            csv_path=tmp_path / "a.csv",
        )
        assert (tmp_path / "r.json").exists()
        assert (tmp_path / "a.csv").exists()


class TestCsvImpliesSaveAttempts:
    def test_passing_csv_keeps_the_records(self, tmp_path: Path) -> None:
        # --csv is useless without the per-request records, so the CLI turns
        # --save-attempts on rather than failing after the run finishes.
        result = runner.invoke(
            app,
            [
                "run",
                "http://127.0.0.1:1/",
                "-n",
                "2",
                "-c",
                "1",
                "-t",
                "2s",
                "--csv",
                str(tmp_path / "a.csv"),
                "--quiet",
            ],
        )
        # The target refuses connections, so the run fails, but the CSV of
        # failures still has to land.
        assert result.exit_code == ExitCode.RUN_FAILED
        assert (tmp_path / "a.csv").exists()


DEAD = "http://127.0.0.1:1/"
FAST = ["-n", "2", "-c", "1", "-t", "1s", "--quiet"]
ONE_REQUEST = ["-n", "1", "-c", "1", "-t", "5s", "--quiet"]


@dataclass(slots=True)
class Live:
    """A target that answers, and every request it saw as method, path, body."""

    url: str
    seen: list[tuple[str, str, str]]


@pytest.fixture
def live() -> Iterator[Live]:
    """Serve 200 OK on a loopback port and record what arrives.

    The CLI opens its own event loop, so a test that drives it stays
    synchronous. A thread keeps this server clear of that loop.
    """
    seen: list[tuple[str, str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def answer(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            seen.append((self.command, self.path, self.rfile.read(length).decode("utf-8")))
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self.answer()

        def do_POST(self) -> None:
            self.answer()

        def log_message(self, format: str, *args: object) -> None:
            """Stay quiet. The default writes one stderr line per request."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Live(url=f"http://127.0.0.1:{server.server_port}/", seen=seen)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _report_file(path: Path, *, success: float, p95: float) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "testbuster/report/1",
                "summary": {"success_rate_pct": success, "requests_per_second": 100.0},
                "latency": {"p50": p95 / 2, "p95": p95, "p99": p95, "p99_9": p95},
            }
        ),
        encoding="utf-8",
    )


class TestDiffCommand:
    def test_no_regression_exits_zero(self, tmp_path: Path) -> None:
        _report_file(tmp_path / "a.json", success=100, p95=50)
        _report_file(tmp_path / "b.json", success=100, p95=50)
        result = runner.invoke(app, ["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
        assert result.exit_code == ExitCode.OK

    def test_a_regression_exits_four(self, tmp_path: Path) -> None:
        _report_file(tmp_path / "a.json", success=100, p95=50)
        _report_file(tmp_path / "b.json", success=100, p95=200)
        result = runner.invoke(
            app, ["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--tolerance", "10"]
        )
        assert result.exit_code == ExitCode.GATE_FAILED

    def test_json_output(self, tmp_path: Path) -> None:
        _report_file(tmp_path / "a.json", success=100, p95=50)
        _report_file(tmp_path / "b.json", success=90, p95=50)
        result = runner.invoke(
            app, ["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--json"]
        )
        payload = json.loads(result.stdout)
        assert any(m["name"] == "success rate" for m in payload["metrics"])

    def test_a_foreign_file_exits_two(self, tmp_path: Path) -> None:
        (tmp_path / "x.json").write_text('{"nope": true}', encoding="utf-8")
        _report_file(tmp_path / "b.json", success=100, p95=50)
        result = runner.invoke(app, ["diff", str(tmp_path / "x.json"), str(tmp_path / "b.json")])
        assert result.exit_code == ExitCode.BAD_USAGE


class TestClusterCommand:
    def test_no_worker_exits_two(self) -> None:
        result = runner.invoke(app, ["cluster", "http://example.com", "-n", "10"])
        assert result.exit_code == ExitCode.BAD_USAGE
        # The words live in cluster.py, so both places say the same thing.
        assert NO_WORKERS in _printed(result)

    def test_help_describes_the_shared_options(self) -> None:
        # cluster reuses the run declarations, so its options carry help text.
        result = runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0
        # Rich wraps a description over lines, so compare on single spaces.
        assert "Skip TLS verification." in " ".join(result.stdout.split())

    def test_all_workers_down_exits_three(self) -> None:
        result = runner.invoke(
            app, ["cluster", "http://example.com", "-w", "http://127.0.0.1:2", "-n", "4"]
        )
        assert result.exit_code == ExitCode.RUN_FAILED


class TestRunOutputs:
    def test_writes_html_and_prometheus_even_on_a_failed_run(self, tmp_path: Path) -> None:
        html = tmp_path / "r.html"
        prom = tmp_path / "r.prom"
        result = runner.invoke(
            app, ["run", DEAD, *FAST, "--html", str(html), "--prometheus", str(prom)]
        )
        # The target refuses every connection, so the run fails.
        assert result.exit_code == ExitCode.RUN_FAILED
        assert html.exists() and html.read_text(encoding="utf-8").startswith("<!doctype html>")
        assert prom.exists() and "testbuster_requests_total" in prom.read_text(encoding="utf-8")

    def test_ndjson_streams_every_attempt(self, tmp_path: Path) -> None:
        out = tmp_path / "a.ndjson"
        runner.invoke(app, ["run", DEAD, *FAST, "--ndjson", str(out)])
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["status"] == 0

    def test_compact_memory_runs(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", DEAD, *FAST, "--compact-memory", "--json"])
        # A failed run still exits 3, but it must produce a parseable report.
        assert result.exit_code == ExitCode.RUN_FAILED
        payload = json.loads(result.stdout)
        assert payload["plan"]["compact_memory"] is True


class TestRunUsageErrors:
    def test_profile_without_duration_exits_two(self) -> None:
        result = runner.invoke(
            app, ["run", DEAD, "--profile", "ramp", "--profile-peak", "50", *FAST]
        )
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_profile_without_peak_exits_two(self) -> None:
        result = runner.invoke(app, ["run", DEAD, "--profile", "ramp", "-D", "1s", "--quiet"])
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_unknown_profile_exits_two(self) -> None:
        result = runner.invoke(
            app, ["run", DEAD, "--profile", "wobble", "--profile-peak", "5", "-D", "1s", "--quiet"]
        )
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_bad_max_latency_exits_two(self) -> None:
        result = runner.invoke(app, ["run", DEAD, *FAST, "--max-latency", "soon"])
        assert result.exit_code == ExitCode.BAD_USAGE


class TestScenarioOption:
    def test_runs_a_scenario_file(self, tmp_path: Path) -> None:
        scenario = tmp_path / "s.json"
        scenario.write_text(
            json.dumps({"steps": [{"name": "dead", "url": DEAD}]}), encoding="utf-8"
        )
        result = runner.invoke(
            app, ["run", "--scenario", str(scenario), "-n", "2", "-c", "1", "-t", "1s", "--quiet"]
        )
        # Every request fails against the closed port, so the run fails.
        assert result.exit_code == ExitCode.RUN_FAILED

    def test_a_missing_step_url_exits_two(self, tmp_path: Path) -> None:
        scenario = tmp_path / "s.json"
        scenario.write_text(json.dumps({"steps": [{"name": "x"}]}), encoding="utf-8")
        result = runner.invoke(app, ["run", "--scenario", str(scenario), "-n", "2", "--quiet"])
        assert result.exit_code == ExitCode.BAD_USAGE

    def test_a_scenario_needs_no_url(self, tmp_path: Path) -> None:
        # The first step supplies the target, so the positional URL is optional.
        scenario = tmp_path / "s.json"
        scenario.write_text(json.dumps({"steps": [{"url": DEAD}]}), encoding="utf-8")
        result = runner.invoke(
            app, ["run", "--scenario", str(scenario), "-n", "1", "-t", "1s", "--quiet"]
        )
        assert result.exit_code == ExitCode.RUN_FAILED

    def test_a_scenario_accepts_every_run_option(self, tmp_path: Path) -> None:
        # Folding scenario into run means gates and outputs work here too, which
        # the separate command did not support.
        scenario = tmp_path / "s.json"
        scenario.write_text(json.dumps({"steps": [{"url": DEAD}]}), encoding="utf-8")
        report = tmp_path / "r.json"
        result = runner.invoke(
            app,
            [
                "run",
                "--scenario",
                str(scenario),
                "-n",
                "2",
                "-t",
                "1s",
                "--max-errors",
                "1",
                "--output",
                str(report),
                "--quiet",
            ],
        )
        assert result.exit_code == ExitCode.RUN_FAILED
        assert report.exists()


class TestOneRequestSource:
    def test_a_scenario_with_a_data_file_exits_two(self, tmp_path: Path) -> None:
        # Both flags name the requests to send. The CSV used to replace the
        # scenario without a word, and the run then sent neither.
        scenario = tmp_path / "s.json"
        scenario.write_text(json.dumps({"steps": [{"url": DEAD}]}), encoding="utf-8")
        rows = tmp_path / "rows.csv"
        rows.write_text("path\nfirst\n", encoding="utf-8")

        result = runner.invoke(
            app, ["run", "--scenario", str(scenario), "--data-file", str(rows), *FAST]
        )
        assert result.exit_code == ExitCode.BAD_USAGE
        assert "Pick one" in _printed(result)


class TestScenarioTakesTheRunFlags:
    """A scenario used to ignore -X, -d, and every --expect-* flag."""

    @staticmethod
    def written(tmp_path: Path, step: dict[str, object]) -> str:
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"steps": [step]}), encoding="utf-8")
        return str(path)

    def test_expect_status_fails_every_step(self, tmp_path: Path, live: Live) -> None:
        scenario = self.written(tmp_path, {"url": live.url + "ok"})
        args = ["run", "--scenario", scenario, "-n", "2", "-c", "1", "-t", "5s", "--quiet"]

        # The same run passes without the check, so the check alone fails it.
        assert runner.invoke(app, args).exit_code == ExitCode.OK

        checked = runner.invoke(app, [*args, "--json", "--expect-status", "500"])
        assert checked.exit_code == ExitCode.RUN_FAILED
        failures = json.loads(checked.stdout)["failures"]
        assert any("not expected" in reason for reason in failures)

    def test_the_method_and_the_body_reach_a_step(self, tmp_path: Path, live: Live) -> None:
        scenario = self.written(tmp_path, {"url": live.url + "echo"})
        result = runner.invoke(
            app, ["run", "--scenario", scenario, "-X", "POST", "-d", "hello", *ONE_REQUEST]
        )
        assert result.exit_code == ExitCode.OK
        assert live.seen == [("POST", "/echo", "hello")]

    def test_a_step_keeps_its_own_method(self, tmp_path: Path, live: Live) -> None:
        # A flag fills a gap in the file. It does not overrule what the step says.
        scenario = self.written(tmp_path, {"url": live.url + "echo", "method": "GET"})
        result = runner.invoke(app, ["run", "--scenario", scenario, "-X", "POST", *ONE_REQUEST])
        assert result.exit_code == ExitCode.OK
        assert live.seen == [("GET", "/echo", "")]


class TestDataFileOption:
    def test_each_row_drives_its_own_url(self, tmp_path: Path, live: Live) -> None:
        rows = tmp_path / "rows.csv"
        rows.write_text("path\nfirst\nsecond\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "run",
                live.url + "{{path}}",
                "--data-file",
                str(rows),
                "-n",
                "2",
                "-c",
                "1",
                "-t",
                "5s",
                "--quiet",
            ],
        )
        assert result.exit_code == ExitCode.OK
        assert [path for _method, path, _body in live.seen] == ["/first", "/second"]

    def test_the_method_and_the_body_reach_a_row(self, tmp_path: Path, live: Live) -> None:
        rows = tmp_path / "rows.csv"
        rows.write_text("name\nada\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "run",
                live.url + "echo",
                "--data-file",
                str(rows),
                "-X",
                "post",
                "-d",
                "hi {{name}}",
                *ONE_REQUEST,
            ],
        )
        assert result.exit_code == ExitCode.OK
        # A lower-case -X still goes out as POST.
        assert live.seen == [("POST", "/echo", "hi ada")]

    def test_a_missing_column_exits_two(self, tmp_path: Path) -> None:
        rows = tmp_path / "rows.csv"
        rows.write_text("path\nfirst\n", encoding="utf-8")
        result = runner.invoke(app, ["run", DEAD + "{{nope}}", "--data-file", str(rows), *FAST])
        assert result.exit_code == ExitCode.BAD_USAGE


class TestScenarioHelp:
    def test_prints_the_run_help_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        # This used to fail: main() fed --help to --scenario as a file name.
        with pytest.raises(SystemExit) as exit_info:
            main(["scenario", "--help"])
        assert exit_info.value.code == 0
        assert "--scenario" in capsys.readouterr().out


class TestWorkerHelp:
    def test_worker_help_lists_host_and_port(self) -> None:
        result = runner.invoke(app, ["worker", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
