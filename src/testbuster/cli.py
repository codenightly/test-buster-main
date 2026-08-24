"""The Test Buster! command line.

Output split: stdout carries the report and nothing else, so `--json` pipes
cleanly into jq. The banner, the run header, progress, and every warning go to
stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Typer 0.27 dropped its click dependency and vendored the completion machinery.
# These two names live in a private module, so tests/test_cli.py generates a
# script for every shell. A typer upgrade that moves them breaks CI loudly
# instead of breaking completion quietly for users.
from typer._completion_shared import Shells, get_completion_script

from testbuster import APP_NAME, COMMAND_NAME, __version__, html_report
from testbuster import cluster as cluster_mod
from testbuster.cluster import NO_WORKERS
from testbuster.config import (
    DEFAULT_RETRY_STATUSES,
    Gates,
    LoadProfile,
    RunPlan,
    normalize_target,
    parse_duration,
    parse_header_json,
    parse_header_pairs,
    resolve_payload,
)
from testbuster.diff import compare, find_regressions, load_report
from testbuster.diff import exit_code as diff_exit_code
from testbuster.engine import Progress as Snapshot
from testbuster.engine import execute, loop_backend
from testbuster.errors import ExitCode, TestBusterError
from testbuster.metrics import Report
from testbuster.reporting import (
    NdjsonWriter,
    print_banner,
    render_header,
    render_report,
    report_to_json,
    write_csv,
    write_json,
    write_prometheus,
)
from testbuster.sources import RequestSource, from_rows, load_rows, load_scenario
from testbuster.validation import build_expectations

_HELP = f"""{APP_NAME} - an async HTTP load generator.

Give it a URL and it sends requests, measures every phase, and prints
percentiles. Add a --fail-* threshold to turn a run into a build step.
"""

_EPILOG = f"""Commands: [bold]{COMMAND_NAME} run URL[/],
[bold]{COMMAND_NAME} diff A B[/], [bold]{COMMAND_NAME} cluster[/], [bold]{COMMAND_NAME} worker[/].

Exit codes: 0 pass, 2 bad usage, 3 the run produced nothing usable,
4 a --fail-* gate failed, 130 stopped with Ctrl+C.
"""

app = typer.Typer(
    name=COMMAND_NAME,
    help=_HELP,
    add_completion=False,  # the completion subcommand below replaces this
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

#: Names that main() must not rewrite into a `run` invocation.
_SUBCOMMANDS: Final[frozenset[str]] = frozenset(
    {"run", "diff", "cluster", "worker", "version", "completion"}
)

#: Flags that ask for help instead of naming a target.
_HELP_FLAGS: Final[frozenset[str]] = frozenset({"-h", "--help"})

#: Environment variable the generated completion scripts set. Typer reads it.
COMPLETE_VAR: Final[str] = "_TESTBUSTER_COMPLETE"


def _panel(name: str) -> Callable[..., Any]:
    """Return a typer.Option factory bound to one help panel.

    Every option in a group shares its panel, so binding it once keeps each
    declaration to a single readable line.
    """

    def make(*names: str, help: str, **extra: Any) -> Any:
        return typer.Option(*names, help=help, rich_help_panel=name, **extra)

    return make


_req = _panel("Request")
_load = _panel("Load shape")
_net = _panel("Network")
_retry = _panel("Retries")
_check = _panel("Response checks")
_gate = _panel("CI gates")
_out = _panel("Output")

# Options that both `run` and `cluster` take. One declaration per option keeps
# the flag names and the help text the same in both commands.
Concurrency = Annotated[
    int,
    _load(
        "--concurrency",
        "--concurrent",
        "-c",
        help="Worker count. This is the ceiling on requests in flight.",
    ),
]
Requests = Annotated[
    int | None,
    _load(
        "--requests",
        "-n",
        help="Stop after this many requests. Defaults to 100 unless --duration is set.",
    ),
]
Duration = Annotated[
    str | None,
    _load(
        "--duration",
        "-D",
        help="Stop after this long: 30s, 5m, 1h30m. Combine with -n to cap both.",
    ),
]
Timeout = Annotated[str, _net("--timeout", "-t", help="Per-request timeout.")]
Insecure = Annotated[bool, _net("--insecure", "-k", help="Skip TLS verification.")]
Output = Annotated[Path | None, _out("--output", "-o", help="Write the JSON report here.")]
AsJson = Annotated[bool, _out("--json", help="Print the JSON report to stdout and nothing else.")]
NoColor = Annotated[bool, _out("--no-color", help="Disable color and styling.")]


def _consoles(no_color: bool) -> tuple[Console, Console]:
    """Return the stdout and stderr consoles."""
    out = Console(file=sys.stdout, no_color=no_color, soft_wrap=False)
    err = Console(file=sys.stderr, no_color=no_color, stderr=True)
    return out, err


def merge_headers(values: list[str]) -> dict[str, str]:
    """Read every -H value into one header dict.

    A value that starts with "{" parses as a JSON object. Anything else parses
    as "Name: value". This accepts both the JSON form and the curl form without
    a second flag.
    """
    merged: dict[str, str] = {}
    for value in values:
        stripped = value.strip()
        if stripped.startswith("{"):
            merged.update(parse_header_json(stripped))
        else:
            merged.update(parse_header_pairs([stripped]))
    return merged


@contextlib.contextmanager
def live_progress(console: Console, plan: RunPlan) -> Iterator[Callable[[Snapshot], None]]:
    """Show one live progress line and yield the callback that drives it.

    A single task owns the display. Rich redraws on its own schedule, so the
    engine only pushes numbers.
    """
    counting_requests = plan.total_requests is not None

    if counting_requests:
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        total: float | None = float(plan.total_requests or 0)
    else:
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ]
        total = plan.duration

    progress = Progress(*columns, console=console, transient=True)

    with progress:
        task = progress.add_task("firing", total=total)

        def push(snapshot: Snapshot) -> None:
            done = snapshot.finished if counting_requests else min(snapshot.elapsed, total or 0.0)
            progress.update(
                task,
                completed=done,
                description=f"firing [green]{snapshot.succeeded}[/] ok "
                f"[red]{snapshot.failed}[/] failed",
            )

        yield push


def _profile_from(
    kind: str | None,
    duration: float | None,
    start: float,
    peak: float | None,
    steps: int,
    spike_at: float,
) -> LoadProfile | None:
    """Build a LoadProfile from the profile flags, or None when unused."""
    if kind is None:
        return None
    if duration is None:
        raise TestBusterError("--profile needs --duration to set its length")
    if peak is None:
        raise TestBusterError("--profile needs --profile-peak, the top rate in req/s")
    return LoadProfile(
        kind=kind,
        duration=duration,
        start_rate=start,
        peak_rate=peak,
        steps=steps,
        spike_at=spike_at,
    )


def _resolve_stop(requests: int | None, duration: float | None) -> int | None:
    """Return the request cap, defaulting to 100 only when nothing else stops."""
    if duration is not None and requests is None:
        return None
    if requests is None:
        return 100
    return requests


def _resolve_target(raw: str | None) -> str:
    """Return the URL to hit, or say what to type when the command line has none.

    A scenario fills in the URL from its first step, so only the other paths
    fail here.
    """
    if not raw:
        raise TestBusterError(f"no target URL. Try: [bold]{COMMAND_NAME} https://example.com[/]")
    return normalize_target(raw)


def _emit(
    report: Report,
    *,
    out: Console,
    err: Console,
    as_json: bool,
    json_path: Path | None,
    csv_path: Path | None,
    html_path: Path | None,
    prometheus_path: Path | None,
) -> ExitCode:
    """Write every requested output and decide the exit code."""
    if as_json:
        # Write it raw. Rich would recolor and rewrap the document, and this
        # output exists to be parsed, not read.
        sys.stdout.write(report_to_json(report) + "\n")
        sys.stdout.flush()
    else:
        render_report(out, report)

    if json_path is not None:
        write_json(report, json_path)
        err.print(f"[dim]json report ->[/] {json_path}")
    if csv_path is not None:
        write_csv(report, csv_path)
        err.print(f"[dim]csv of attempts ->[/] {csv_path}")
    if html_path is not None:
        html_report.write(report, html_path, tool_version=__version__)
        err.print(f"[dim]html report ->[/] {html_path}")
    if prometheus_path is not None:
        write_prometheus(report, prometheus_path)
        err.print(f"[dim]prometheus metrics ->[/] {prometheus_path}")

    # A run that measured nothing, or where every single request failed, is not
    # a pass. Reporting success there would hide a dead target from CI.
    if report.total == 0 or report.succeeded == 0:
        return ExitCode.RUN_FAILED

    if report.stop_reason == "max errors reached":
        err.print("[bold red]run failed:[/] the failure cap was reached")
        return ExitCode.RUN_FAILED

    failed_gates = [gate for gate in report.check_gates() if not gate.passed]
    if failed_gates:
        names = ", ".join(gate.name for gate in failed_gates)
        err.print(f"[bold red]gate failed:[/] {names}")
        return ExitCode.GATE_FAILED

    if report.interrupted:
        return ExitCode.INTERRUPTED
    return ExitCode.OK


@app.command("run", epilog=_EPILOG, no_args_is_help=True)
def run_command(
    url: Annotated[
        str | None,
        typer.Argument(metavar="URL", help="Target URL. A bare host gets https:// added."),
    ] = None,
    target_flag: Annotated[str | None, _req("--url", "-u", help="Target URL, as a flag.")] = None,
    method: Annotated[str, _req("--method", "-X", help="HTTP method.")] = "GET",
    header: Annotated[
        list[str] | None,
        _req("--header", "-H", help="Header as 'Name: value', or a JSON object. Repeatable."),
    ] = None,
    body: Annotated[
        str | None,
        _req("--body", "-d", help="Request body. Prefix with @ to read a file, like curl."),
    ] = None,
    data_file: Annotated[
        Path | None,
        _req(
            "--data-file",
            help="CSV of values for {{column}} placeholders in the URL, headers, and body.",
        ),
    ] = None,
    scenario: Annotated[
        Path | None,
        _req(
            "--scenario",
            help="Weighted request steps from a JSON or YAML file. Replaces the URL.",
        ),
    ] = None,
    concurrency: Concurrency = 10,
    requests: Requests = None,
    duration: Duration = None,
    rate: Annotated[
        float | None, _load("--rate", help="Cap requests per second across all workers.")
    ] = None,
    max_errors: Annotated[
        int | None, _load("--max-errors", help="Stop the run once this many requests have failed.")
    ] = None,
    warmup: Annotated[
        int,
        _load(
            "--warmup", help="Send this many throwaway requests first. They stay out of the report."
        ),
    ] = 0,
    profile: Annotated[
        str | None,
        _load(
            "--profile",
            help="Vary the rate over time: constant, ramp, step, or spike. Needs --duration.",
        ),
    ] = None,
    profile_start: Annotated[
        float, _load("--profile-start", help="Starting rate for a profile.")
    ] = 1.0,
    profile_peak: Annotated[
        float | None, _load("--profile-peak", help="Top rate for a profile, in req/s.")
    ] = None,
    profile_steps: Annotated[
        int, _load("--profile-steps", help="Number of levels for the step profile.")
    ] = 4,
    spike_at: Annotated[float, _load("--spike-at", help="Where a spike peaks, from 0 to 1.")] = 0.5,
    timeout: Timeout = "30s",
    insecure: Insecure = False,
    proxy: Annotated[
        str | None, _net("--proxy", "-p", help="Proxy URL. SOCKS needs the socks extra.")
    ] = None,
    http2: Annotated[
        bool, _net("--http2", help="Use HTTP/2 through httpx. Needs the http2 extra.")
    ] = False,
    cookies: Annotated[
        bool, _net("--cookies", help="Keep cookies across the run, for a login flow.")
    ] = False,
    follow_redirects: Annotated[
        bool, _net("--follow-redirects/--no-follow-redirects", help="Follow 3xx responses.")
    ] = True,
    keepalive: Annotated[
        bool,
        _net(
            "--keepalive/--no-keepalive",
            help="Reuse connections. Turn it off to measure handshake cost every time.",
        ),
    ] = True,
    retries: Annotated[int, _retry("--retries", help="Extra tries per failed request.")] = 0,
    retry_backoff: Annotated[
        str, _retry("--retry-backoff", help="Base backoff. It doubles per try and carries jitter.")
    ] = "250ms",
    retry_on: Annotated[
        list[int] | None,
        _retry(
            "--retry-on",
            help="Status codes worth retrying. Repeatable. Defaults to 408, 425, 429, 5xx.",
        ),
    ] = None,
    expect_status: Annotated[
        list[str] | None,
        _check(
            "--expect-status",
            help="A response outside these codes is a failure: 200, 2xx, 200-204. Repeatable.",
        ),
    ] = None,
    expect_regex: Annotated[
        str | None,
        _check("--expect-regex", help="Fail a response whose body does not match this pattern."),
    ] = None,
    expect_json: Annotated[
        list[str] | None,
        _check(
            "--expect-json",
            help="Fail a response where a JSON path is not the value: data.id=42. Repeatable.",
        ),
    ] = None,
    max_latency: Annotated[
        str | None,
        _check("--max-latency", help="Fail any single request slower than this: 500ms, 2s."),
    ] = None,
    max_p95: Annotated[
        float | None,
        _gate("--fail-over-p95", help="Exit 4 when p95 latency in ms goes above this."),
    ] = None,
    max_p99: Annotated[
        float | None,
        _gate("--fail-over-p99", help="Exit 4 when p99 latency in ms goes above this."),
    ] = None,
    min_success: Annotated[
        float | None,
        _gate(
            "--fail-under-success", help="Exit 4 when the success rate drops below this percentage."
        ),
    ] = None,
    output: Output = None,
    html: Annotated[
        Path | None, _out("--html", help="Write a self-contained HTML report here.")
    ] = None,
    prometheus_out: Annotated[
        Path | None, _out("--prometheus", help="Write Prometheus text metrics here.")
    ] = None,
    csv_output: Annotated[
        Path | None, _out("--csv", help="Write one CSV row per request. Turns on --save-attempts.")
    ] = None,
    ndjson: Annotated[
        str | None,
        _out("--ndjson", help="Stream one JSON line per request to a file, or '-' for stdout."),
    ] = None,
    save_attempts: Annotated[
        bool,
        _out("--save-attempts", help="Keep every per-request record. Costs memory on long runs."),
    ] = False,
    compact_memory: Annotated[
        bool,
        _out(
            "--compact-memory",
            help="Skip the exact sample arrays. Percentiles come from the histogram.",
        ),
    ] = False,
    as_json: AsJson = False,
    quiet: Annotated[
        bool, _out("--quiet", "-q", help="Hide the banner, header, and progress.")
    ] = False,
    no_banner: Annotated[bool, _out("--no-banner", help="Hide the banner only.")] = False,
    no_color: NoColor = False,
) -> None:
    """Send load at a URL and report what happened.

    A bare `testbuster URL` lands here, so the `run` word is optional.
    """
    out, err = _consoles(no_color)
    silent = quiet or as_json

    try:
        # Refuse the pair before either source is built. A run sends one set of
        # requests, and the second flag used to replace the first in silence.
        if scenario is not None and data_file is not None:
            raise TestBusterError("--scenario and --data-file both supply the requests. Pick one")

        # A spec built before the plan carries the method as typed, and "post"
        # is not "POST" on the wire.
        method = method.strip().upper()
        headers = merge_headers(header or [])
        # Read the body once. A "@file" body would otherwise load twice.
        payload = resolve_payload(body)
        parsed_duration = parse_duration(duration, label="--duration") if duration else None
        expectations = build_expectations(
            status=expect_status,
            body_regex=expect_regex,
            json_specs=expect_json,
            max_latency_s=(
                parse_duration(max_latency, label="--max-latency") if max_latency else None
            ),
        )

        source: RequestSource | None = None
        if scenario is not None:
            # The flags fill in what a step leaves out, so -X, -d, and the
            # --expect-* checks reach every step.
            source = load_scenario(
                scenario,
                default_method=method,
                default_body=payload,
                default_expectations=expectations,
            )
            # A scenario carries its own URLs, so it stands in for the target.
            # The plan still needs one for the connector and TLS settings, so
            # the first step supplies it.
            url = url or source.spec_for(0).url
        elif data_file is not None:
            source = from_rows(
                load_rows(data_file),
                method=method,
                url_template=_resolve_target(target_flag or url),
                header_templates=headers,
                body_template=payload,
                expectations=expectations,
            )

        plan = RunPlan(
            target=_resolve_target(target_flag or url),
            method=method,
            headers=headers,
            payload=payload,
            workers=concurrency,
            total_requests=_resolve_stop(requests, parsed_duration),
            duration=parsed_duration,
            rate_limit=rate,
            max_errors=max_errors,
            warmup=warmup,
            profile=_profile_from(
                profile, parsed_duration, profile_start, profile_peak, profile_steps, spike_at
            ),
            timeout=parse_duration(timeout, label="--timeout"),
            verify_tls=not insecure,
            follow_redirects=follow_redirects,
            keepalive=keepalive,
            proxy=proxy,
            http2=http2,
            cookies=cookies,
            retries=retries,
            retry_backoff=parse_duration(retry_backoff, label="--retry-backoff"),
            retry_statuses=frozenset(retry_on) if retry_on else frozenset(DEFAULT_RETRY_STATUSES),
            expectations=expectations,
            compact_memory=compact_memory,
            keep_attempts=save_attempts or csv_output is not None,
            gates=Gates(max_p95_ms=max_p95, max_p99_ms=max_p99, min_success_rate=min_success),
        )

        if not silent and not no_banner:
            print_banner(err)
        if not silent:
            render_header(err, plan, loop_backend())

        writer = NdjsonWriter(ndjson) if ndjson else None
        on_attempt = writer.write if writer else None
        try:
            if silent:
                report = execute(plan, source=source, on_attempt=on_attempt)
            else:
                with live_progress(err, plan) as push:
                    report = execute(plan, push, source=source, on_attempt=on_attempt)
        finally:
            if writer is not None:
                writer.close()

        code = _emit(
            report,
            out=out,
            err=err,
            as_json=as_json,
            json_path=output,
            csv_path=csv_output,
            html_path=html,
            prometheus_path=prometheus_out,
        )
        raise typer.Exit(int(code))

    except TestBusterError as exc:
        err.print(f"[bold red]error[/] {exc}")
        raise typer.Exit(int(exc.code)) from exc


@app.command("diff", no_args_is_help=True)
def diff_command(
    base: Annotated[Path, typer.Argument(help="The baseline JSON report.")],
    new: Annotated[Path, typer.Argument(help="The new JSON report to compare.")],
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance", help="Allowed regression, as a percent. A move worse than this exits 4."
        ),
    ] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the deltas as JSON.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """Compare two saved reports and flag regressions."""
    out, err = _consoles(no_color)
    try:
        deltas = compare(load_report(base), load_report(new))
        regressions = find_regressions(deltas, tolerance)

        if as_json:
            payload = {
                "tolerance_percent": tolerance,
                "metrics": [
                    {
                        "name": d.name,
                        "unit": d.unit,
                        "base": d.base,
                        "new": d.new,
                        "percent": round(d.percent, 3),
                        "regression": d.regressed_beyond(tolerance),
                    }
                    for d in deltas
                ],
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            table = Table(title="report diff", title_justify="left", title_style="bold")
            table.add_column("metric", style="dim")
            table.add_column("base", justify="right")
            table.add_column("new", justify="right")
            table.add_column("change", justify="right")
            for delta in deltas:
                worse = delta.regressed_beyond(tolerance)
                # ASCII only, so an old Windows code page cannot crash the table.
                style = "red" if worse else ("dim" if delta.is_regression else "green")
                table.add_row(
                    delta.name,
                    f"{delta.base:.2f} {delta.unit}",
                    f"{delta.new:.2f} {delta.unit}",
                    f"[{style}]{delta.percent:+.1f}%[/]",
                )
            out.print(table)
            if regressions:
                names = ", ".join(r.name for r in regressions)
                err.print(f"[bold red]regressed:[/] {names}")

        raise typer.Exit(int(diff_exit_code(regressions)))

    except TestBusterError as exc:
        err.print(f"[bold red]error[/] {exc}")
        raise typer.Exit(int(exc.code)) from exc


@app.command("worker")
def worker_command(
    host: Annotated[str, typer.Option("--host", help="Address to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8637,
) -> None:
    """Start a worker service for a distributed run. This blocks until stopped."""
    err = Console(file=sys.stderr, stderr=True)
    err.print(f"[bold cyan]{APP_NAME}[/] worker listening on http://{host}:{port}")
    err.print("[dim]point a cluster run at this with --worker[/]")
    cluster_mod.run_worker(host, port)


@app.command("cluster", no_args_is_help=True)
def cluster_command(
    url: Annotated[str, typer.Argument(metavar="URL", help="Target URL for every worker.")],
    worker: Annotated[
        list[str] | None, typer.Option("--worker", "-w", help="A worker base URL. Repeatable.")
    ] = None,
    concurrency: Concurrency = 10,
    requests: Requests = None,
    duration: Duration = None,
    timeout: Timeout = "30s",
    insecure: Insecure = False,
    output: Output = None,
    as_json: AsJson = False,
    no_color: NoColor = False,
) -> None:
    """Split a run across worker services and merge their reports."""
    out, err = _consoles(no_color)
    try:
        if not worker:
            raise TestBusterError(NO_WORKERS)

        parsed_duration = parse_duration(duration, label="--duration") if duration else None
        plan = RunPlan(
            target=url,
            workers=concurrency,
            total_requests=_resolve_stop(requests, parsed_duration),
            duration=parsed_duration,
            timeout=parse_duration(timeout, label="--timeout"),
            verify_tls=not insecure,
        )

        err.print(f"[dim]dispatching to {len(worker)} worker(s)[/]")
        merged = asyncio.run(cluster_mod.dispatch(plan, worker))
        summary = merged["summary"]

        if as_json:
            # One compact line, because this output exists to be parsed.
            sys.stdout.write(json.dumps(merged) + "\n")
        else:
            out.print(f"[bold]merged from {merged.get('merged_from', 0)} workers[/]")
            out.print(f"requests    {summary['total_requests']:,}")
            out.print(f"success     {summary['success_rate_pct']:.2f}%")
            out.print(f"throughput  {summary['requests_per_second']:,.1f} req/s")
            out.print(f"p95 latency {merged['latency']['p95']:.2f} ms")

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            err.print(f"[dim]merged report ->[/] {output}")

        failed = summary["total_requests"] == 0 or summary["successful"] == 0
        raise typer.Exit(int(ExitCode.RUN_FAILED if failed else ExitCode.OK))

    except TestBusterError as exc:
        err.print(f"[bold red]error[/] {exc}")
        raise typer.Exit(int(exc.code)) from exc


@app.command("version")
def version_command(
    plain: Annotated[bool, typer.Option("--plain", help="Print only the version number.")] = False,
) -> None:
    """Show the version."""
    out = Console(file=sys.stdout)
    if plain:
        out.print(__version__, highlight=False)
        return

    err = Console(file=sys.stderr, stderr=True)
    print_banner(err)
    out.print(f"{APP_NAME} {__version__}")
    out.print(f"python {sys.version.split()[0]} on {sys.platform}")
    out.print(f"event loop: {loop_backend()}")


@app.command("completion")
def completion_command(
    shell: Annotated[Shells, typer.Argument(help="The shell to generate a script for.")],
) -> None:
    """Print a shell completion script.

    Send the output to the location your shell reads. The README lists the path
    for each shell.
    """
    script = get_completion_script(
        prog_name=COMMAND_NAME,
        complete_var=COMPLETE_VAR,
        shell=shell.value,
    )
    sys.stdout.write(script)


def main(argv: list[str] | None = None) -> None:
    """Entry point.

    A positional URL and a subcommand cannot coexist in one Click group: the
    argument eats the subcommand name, so `testbuster version` would load test
    https://version. This inserts the implicit `run` before handing over, which
    keeps both `testbuster URL` and `testbuster version` working.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        args = ["--help"]
    elif args[0] == "scenario":
        # A scenario is a run whose requests come from a file, so it needs no
        # command of its own. This keeps the shorter spelling working, and a
        # missing file lands on typer's own "requires an argument" message.
        rest = args[1:]
        if rest and rest[0] in _HELP_FLAGS:
            # Passed on, the flag would name a file called "--help".
            args = ["run", "--help"]
        else:
            args = ["run", "--scenario", *rest]
    elif args[0] not in _SUBCOMMANDS and args[0] not in _HELP_FLAGS:
        args.insert(0, "run")

    app(args)


if __name__ == "__main__":
    main()
