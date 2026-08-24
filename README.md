# Test Buster!

Async HTTP load generation with per-phase timings, response checks, load
profiles, scenarios, live and saved reports, and CI gates.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)

Point it at a URL. It sends requests, measures where the time went, and prints
percentiles. Add a `--fail-*` threshold and the same command becomes a build
step that fails a pull request when latency regresses.

```bash
testbuster https://api.example.com -n 1000 -c 50
```

> **Note on the name.** The tool is *Test Buster!* and the command is `testbuster`, because a shell command cannot hold a space or an exclamation mark.

## What it measures

Most load testers report one number per request. Test Buster! splits the time
into phases, so a slow p99 points at a cause.

| Series | What it covers |
|---|---|
| **latency** | The whole request: first byte sent to last byte read |
| **ttfb** | Request sent until the response headers arrive |
| **connect** | TCP handshake and TLS handshake. `0` means the pool reused a socket |
| **dns** | Name resolution. Absent when the address was cached or given as an IP |

Each series reports min, mean, stdev, max, and p50/p75/p90/p95/p99/p99.9.

A p99 of 400 ms tells you something is slow. A p99 where `connect` holds 380 ms
of it tells you to look at TLS termination, not the application. The report also
carries a latency histogram, a sparkline of its shape, and a throughput
timeline. A bimodal shape that one p99 would hide shows up.

## Install

The package on PyPI is the recommended route. `pipx` and `uv` each keep the tool
in its own environment.

```bash
pipx install test-buster
uv tool install test-buster

# extras add the optional backends
pipx install 'test-buster[http2]'   # HTTP/2 through httpx
pipx install 'test-buster[socks]'   # SOCKS4 and SOCKS5 proxy support
pipx install 'test-buster[yaml]'    # YAML scenario files (JSON needs no extra)
pipx install 'test-buster[speed]'   # uvloop, for a higher request ceiling
pipx install 'test-buster[all]'     # every runtime extra at once
```

`uvloop` raises the requests-per-second ceiling and does not build on Windows.
Test Buster! uses it when it can import it. The run header names the loop in use.

Install from a checkout to work on the tool. Python 3.11 is the floor, for
`asyncio.TaskGroup` and `asyncio.Runner`.

```bash
git clone https://github.com/your-account/test-buster.git
cd test-buster
python -m pip install -e ".[dev]"
pytest -q
```

Send a completion script to the location your shell reads:

```
testbuster completion bash | sudo tee /etc/bash_completion.d/testbuster
testbuster completion zsh > "${fpath[1]}/_testbuster"
testbuster completion fish > ~/.config/fish/completions/testbuster.fish
testbuster completion powershell | Out-String | Invoke-Expression
```

## Build a standalone executable

`freeze.py` builds one file that runs on a machine with no Python. Install every
extra first, or the binary will lack the optional backends.

```bash
python -m pip install -e ".[all,freeze]"
python freeze.py
```

The result lands in `dist/`, about 14 MiB on Windows. The script verifies it
before it reports success: version, help, completion for four shells, a real
load run against a local server, and each of the four exit codes. A failure
there fails the build.

| Flag | Effect |
|---|---|
| `--backend nuitka` | Compile to C first. Starts faster, needs a C compiler and `".[nuitka]"` |
| `--onedir` | A folder beside the executable. Starts faster, ships as an archive |
| `--tagged` | Name it `testbuster-linux-amd64`, for release assets |
| `--name NAME` | Change the base output name |
| `--no-verify` | Skip the checks |
| `--keep-work` | Leave the intermediate build tree for inspection |

Neither backend links libpython statically. Both embed the CPython runtime, so
the one file needs no interpreter on the target machine.

The script handles two details that fail quietly otherwise. The engine imports
`aiohttp_socks`, `httpx`, `uvloop`, `yaml`, and `aiohttp.web` inside functions,
where a freezer does not find them, so the script passes each one it can import.
The version comes from `importlib.metadata`, which needs the distribution
metadata copied in, or the binary reports `0.0.0+dev`.

## Quick start

```bash
testbuster https://api.example.com                       # 100 requests, 10 workers
testbuster api.example.com                               # a bare host gets https://
testbuster https://api.example.com -n 1000 -c 50 -t 10s  # 1000 requests, 50 at a time
testbuster https://api.example.com -D 30s -c 25          # a fixed span, not a count
testbuster https://api.example.com -D 1m --rate 200      # hold 200 requests per second
testbuster https://api.example.com -n 500 --fail-over-p95 250 --fail-under-success 99.5
```

A POST with headers and a body:

```bash
testbuster https://api.example.com/users -X POST \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer token' \
  -d '{"name": "Ada Lovelace"}'
```

`-H` repeats, and it takes the curl form or a JSON object such as
`-H '{"Accept": "application/json"}'`. A body that starts with `@` names a file,
the same way curl does: `-d @payload.json`.

## Commands

```
testbuster [run] URL [OPTIONS]          # send load at one URL
testbuster run --scenario FILE ...      # run a weighted set of request steps
testbuster diff BASE NEW [OPTIONS]      # compare two saved reports
testbuster worker [--host --port]       # run a worker for a distributed run
testbuster cluster URL --worker U ...   # split a run across workers
testbuster version [--plain]
testbuster completion {bash|zsh|fish|powershell}
```

The `run` word is optional, so `testbuster https://x` and
`testbuster run https://x` are the same command.

## The run command

### Request

| Short | Long | Default | Description |
|---|---|---|---|
| `-u` | `--url` | | Target URL, as a flag instead of an argument |
| `-X` | `--method` | `GET` | GET, HEAD, POST, PUT, PATCH, DELETE, or OPTIONS |
| `-H` | `--header` | | `Name: value`, or a JSON object. Repeatable |
| `-d` | `--body` | | Request body. `@path` reads a file |
| | `--data-file` | | CSV of values for `{{column}}` placeholders |
| | `--scenario` | | Weighted request steps from a JSON or YAML file |

### Load shape

| Short | Long | Default | Description |
|---|---|---|---|
| `-c` | `--concurrency` | `10` | Worker count, the ceiling on requests in flight |
| `-n` | `--requests` | `100` | Stop after this many. The default applies unless `--duration` is set |
| `-D` | `--duration` | | Stop after this long: `30s`, `5m`, `1h30m` |
| | `--rate` | | Cap requests per second across all workers |
| | `--max-errors` | | Stop once this many requests have failed |
| | `--warmup` | `0` | Throwaway requests to send first, excluded from the report |
| | `--profile` | | Vary the rate over time: `constant`, `ramp`, `step`, `spike` |
| | `--profile-start` | `1` | Starting rate for a profile |
| | `--profile-peak` | | Top rate for a profile, in req/s |
| | `--profile-steps` | `4` | Levels for the step profile |
| | `--spike-at` | `0.5` | Where a spike peaks, from 0 to 1 |

`--concurrent` works as an alias for `--concurrency`. Pass `-n` and `-D`
together and the run stops at whichever limit arrives first.

### Network

| Short | Long | Default | Description |
|---|---|---|---|
| `-t` | `--timeout` | `30s` | Per-request timeout |
| `-k` | `--insecure` | off | Skip TLS verification |
| `-p` | `--proxy` | | `http://`, `https://`, `socks4://`, `socks5://`, or `socks5h://` |
| | `--http2` | off | Use HTTP/2 through httpx. Needs the http2 extra |
| | `--cookies` | off | Keep cookies across the run, for a login flow |
| | `--follow-redirects` / `--no-follow-redirects` | on | Follow 3xx responses |
| | `--keepalive` / `--no-keepalive` | on | Reuse connections. Turn it off to price the handshake every time |

### Retries

| Long | Default | Description |
|---|---|---|
| `--retries` | `0` | Extra tries per failed request |
| `--retry-backoff` | `250ms` | Base backoff. It doubles per try and carries jitter |
| `--retry-on` | `408 425 429 500 502 503 504` | Status codes worth retrying. Repeatable |

### Response checks

A status of 200 does not mean the response was right. A failed check lowers the
success rate and can trip a gate.

| Long | Description |
|---|---|
| `--expect-status` | A response outside these codes fails: `200`, `2xx`, `200-204`. Repeatable |
| `--expect-regex` | Fail a response whose body does not match this pattern |
| `--expect-json` | Fail a response where a JSON path is not the value: `data.id=42`. Repeatable |
| `--max-latency` | Fail any single request slower than this: `500ms`, `2s` |

```bash
testbuster https://api.example.com/health --expect-status 200 \
  --expect-json status=ok --max-latency 300ms -n 200
```

A failed check shows up in the failures table with its reason. The response
still counts in the status-code table under its real code.

### CI gates

| Long | Description |
|---|---|
| `--fail-over-p95 MS` | Exit 4 when p95 latency goes above this |
| `--fail-over-p99 MS` | Exit 4 when p99 latency goes above this |
| `--fail-under-success PCT` | Exit 4 when the success rate drops below this |

### Output

| Short | Long | Description |
|---|---|---|
| `-o` | `--output PATH` | Write the JSON report to a file |
| | `--html PATH` | Write a self-contained HTML report with charts |
| | `--prometheus PATH` | Write Prometheus text metrics |
| | `--csv PATH` | One CSV row per request. Turns on `--save-attempts` |
| | `--ndjson PATH` | Stream one JSON line per request. `-` means stdout |
| | `--save-attempts` | Keep every per-request record. Costs memory on long runs |
| | `--compact-memory` | Skip the sample arrays. Percentiles come from the histogram |
| | `--json` | Print the JSON report to stdout and nothing else |
| `-q` | `--quiet` | Hide the banner, header, and progress |
| | `--no-banner` | Hide the banner only |
| | `--no-color` | Disable color and styling |

stdout carries the report and nothing else. The banner, run header, progress
bar, and warnings all go to stderr, so `--json` pipes cleanly:

```bash
testbuster https://api.example.com --json | jq '.latency.p95'
```

## Load profiles

A flat rate never finds the point where a system tips over. A profile varies the
rate. Every profile needs `--duration` and `--profile-peak`.

```bash
# ramp 10 -> 500 req/s over five minutes
testbuster https://api.example.com -D 5m --profile ramp --profile-start 10 --profile-peak 500
# four plateaus, or a spike to 1000 req/s halfway through
testbuster https://api.example.com -D 4m --profile step --profile-peak 400 --profile-steps 4
testbuster https://api.example.com -D 2m --profile spike --profile-peak 1000 --spike-at 0.5
```

## Scenarios

A real user does more than hit one endpoint. A scenario is a weighted list of
steps. Each step carries its own method, URL, headers, body, and checks. The
report breaks the numbers down by step.

```json
{
  "base_url": "https://api.example.com",
  "steps": [
    { "name": "list", "url": "/items", "weight": 5, "expect": { "status": ["2xx"] } },
    { "name": "read", "url": "/items/1", "weight": 3 },
    { "name": "create", "method": "POST", "url": "/items", "weight": 1,
      "body": { "name": "widget" },
      "expect": { "status": ["201"], "json": ["id"], "max_latency_ms": 500 } }
  ]
}
```

A scenario is an option on `run`, not a command of its own. The shorter spelling
also works. `main()` rewrites `scenario FILE ...` into `run --scenario FILE ...`
before the parser sees it.

```bash
testbuster run --scenario flow.json -c 25 -D 2m
testbuster scenario flow.json -c 25 -D 2m       # the same run
```

A scenario accepts every option on this page. Profiles, retries, gates, and each
output format all apply. The weights set the mix exactly. A 5:3:1 mix walks a
repeating wheel with no random draw, so a run repeats. YAML needs the `yaml`
extra.

## Data files

A run that hits one URL warms one cache entry. `--data-file` draws values from a
CSV and fills `{{column}}` placeholders in the URL, the headers, and the body.

```csv
id,token
1,abc
2,def
```

```bash
testbuster "https://api.example.com/users/{{id}}" --data-file users.csv \
  -H 'Authorization: Bearer {{token}}' -n 300
```

The tool builds one request per row at load time. The rows then cycle, so the
row count and the request count do not have to match.

## Diffing two runs

A single run says how a system behaves now. A pair says whether a change made it
better or worse.

```bash
testbuster https://api.example.com -n 500 -o before.json
# ... deploy a change ...
testbuster https://api.example.com -n 500 -o after.json
testbuster diff before.json after.json --tolerance 10
```

Lower latency is better. Higher success rate and throughput are better. A move
in the wrong direction past `--tolerance` percent exits 4, so a diff can gate a
deploy. Add `--json` for a machine-readable delta, or `--no-color` for plain text.

## Distributed runs

One machine caps out before most production systems do. Start a worker on each
load box, then point a coordinator at them.

```bash
testbuster worker --host 0.0.0.0 --port 8637   # on each load box
# then, from the coordinator
testbuster cluster https://api.example.com \
  --worker http://box-1:8637 --worker http://box-2:8637 -n 100000 -c 200
```

The coordinator splits the request count across the workers, collects their
reports, and merges them. Merged percentiles come from the combined histograms,
so they are approximate the same way `--compact-memory` is.

`cluster` takes `--worker`/`-w`, `-c`, `-n`, `-D`, `-t`, `-k`, `-o`, `--json`,
and `--no-color`. Use `run` for anything else, such as a gate or an HTML report.

## Exit codes and CI

| Code | Meaning |
|---|---|
| `0` | The run finished and every gate passed |
| `2` | Bad flags, a bad URL, or a bad method |
| `3` | The run produced nothing usable: no requests completed, every one failed, or `--max-errors` tripped |
| `4` | A `--fail-*` gate or a `diff --tolerance` was missed |
| `130` | Ctrl+C. Partial results still print |

A GitHub Actions step that gates a pull request:

```yaml
- name: Load test the staging API
  run: |
    pipx install test-buster
    testbuster https://staging.example.com/health \
      --duration 30s --concurrency 25 --expect-json status=ok \
      --fail-over-p95 250 --fail-under-success 99.5 \
      --output load-report.json --html load-report.html --no-banner

- uses: actions/upload-artifact@v4
  if: always()
  with: { name: load-report, path: load-report.* }
```

## Output formats

| Format | Flag | What it holds |
|---|---|---|
| Console | the default | Summary, phase tables, sparkline, status codes, failures, steps, gates |
| JSON | `--output PATH`, `--json` | The whole report, under a versioned `schema` key |
| HTML | `--html PATH` | Summary tiles, a histogram, a timeline, and the tables |
| Prometheus | `--prometheus PATH` | The text exposition format. Every metric carries a `target` label |
| CSV | `--csv PATH` | One row per request |
| NDJSON | `--ndjson PATH` | One JSON line per request, streamed during the run |

Percentiles and status codes print in a fixed order, so two reports diff
cleanly. Every duration in the JSON is a number of milliseconds. The JSON also
carries a compact histogram and a throughput timeline. Per-request records appear
under `attempts` only with `--save-attempts`. Credentials in `Authorization`,
`Proxy-Authorization`, `Cookie`, `X-API-Key`, and `API-Key` become `<redacted>`,
so a report is safe to attach to a build. The HTML page pulls in no external
assets. NDJSON takes `-` for stdout, which feeds a pipeline without buffering.

## How the engine works

**A fixed worker pool, not a task per request.** The engine starts exactly
`--concurrency` asyncio tasks. Each one claims the next request from a shared
counter and loops until a stop condition fires. Memory stays flat at any length.

**Requests are built once.** A source holds a list of request specs and a wheel
of positions. A CSV fills its placeholders at load time, not per request.

**Bodies are counted, not kept.** Each response streams through a 64 KiB buffer
that counts bytes and drops them. A body check keeps a bounded copy.

**Two number stores.** Exact percentiles come from a compact `array('d')` of
samples. A log-linear histogram runs alongside it at bounded memory and powers
the HTML charts and the merge. `--compact-memory` drops the array.

**Percentiles interpolate.** The rank falls between two samples. The result is
the weighted value between them, which is what NumPy does by default.

**One task owns the progress line.** Progress redraws on a timer, not from every
worker, so the display cannot shred itself.

**Ctrl+C reports.** The first Ctrl+C sets a stop event. Workers finish the
request in flight and exit. The report prints what was measured, marked as
interrupted, and the process exits `130`. A second Ctrl+C ends it at once.

**A token bucket paces the rate.** The burst allowance is a tenth of a second of
traffic, so a short run cannot ignore `--rate`. A profile updates the bucket
rate as time passes.

Drive the same engine from your own code:

```python
from testbuster.config import Gates, RunPlan
from testbuster.engine import execute

plan = RunPlan(
    target="https://api.example.com", workers=25, total_requests=1000, gates=Gates(max_p95_ms=250)
)
report = execute(plan)
print(report.latency.percentiles_ms["p95"])
print([gate for gate in report.check_gates() if not gate.passed])
```

`RunPlan` validates on construction, so a bad plan raises `TestBusterError`
before the first request goes out. Await `LoadEngine(plan).run()` instead to
drive it inside your own event loop.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Every request fails with `dns lookup failed` | The hostname does not resolve here. Check the spelling and the network |
| Every request fails with `connection refused` | Nothing listens on that port. Try `curl -I` against the same URL first |
| The success rate drops as `--concurrency` rises | The target rate limits, or it runs out of workers. Look for `429`, then add `--rate` or lower `--concurrency` |
| `tls certificate rejected` | `--insecure` skips the check for a self-signed staging box. Do not use it against production |
| Throughput plateaus below what the target can serve | The generator is the bottleneck. Install the speed extra for `uvloop`, raise `--concurrency`, or run `testbuster cluster` across more machines |
| A SOCKS proxy, `--http2`, or a YAML scenario reports a missing extra | Install it with `pipx install 'test-buster[all]'` |
| The banner renders as garbage on Windows | The console code page cannot hold the block characters. Run `chcp 65001`, or pass `--no-banner` |

## Roadmap

Every item on the first roadmap is built, and this page documents each one.
Still open, roughly in order of value:

1. **HTTP/3.** This needs an aioquic transport behind the existing backend switch.
2. **A live full-screen TUI.** A rolling latency chart while the run is in flight.
3. **Per-worker cookie jars.** Simulate many distinct signed-in users, not one session.
4. **OpenTelemetry export.** Push spans and metrics during the run, next to the Prometheus output.
5. **OS packages.** Build DEB and RPM packages from the wheel in the release workflow.

## Contributing

This page is the whole contract for a contribution. Read it before you open a
pull request.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Python 3.11 is the floor. The engine uses `asyncio.TaskGroup` and
`asyncio.Runner`. Both arrived in 3.11.

### The checks

All four must pass before a pull request merges. CI runs the same commands.

```bash
pytest -q                 # 440 tests across 8 files
ruff check . && ruff format --check . && mypy
```

Run `ruff format .` to fix formatting rather than hand-wrapping lines. `mypy`
runs in strict mode over `src/testbuster`. New code needs full annotations.

### Layout

Fourteen modules live in `src/testbuster`.

| Module | Holds |
|---|---|
| `__init__.py` | `APP_NAME`, `COMMAND_NAME`, and `__version__` |
| `__main__.py` | The `python -m testbuster` entry point |
| `errors.py` | `ExitCode` and `TestBusterError` |
| `histogram.py` | The bounded-memory latency histogram, which also merges |
| `validation.py` | Response `Expectations` and their checks |
| `config.py` | `RunPlan`, `Gates`, `LoadProfile`, and every parse helper. Validation lives here |
| `sources.py` | `RequestSpec`, the `RequestSource` protocol, and `RequestCycle` with its factories |
| `metrics.py` | `Attempt`, `Tally`, `Spread`, `LabelStats`, `Report`, and the percentile math |
| `engine.py` | The worker pool, the rate limiter, retries, phase timing, and the two transports |
| `reporting.py` | The banner, console tables, and the JSON, CSV, NDJSON, and Prometheus writers |
| `html_report.py` | The self-contained HTML report |
| `diff.py` | Comparing two saved reports |
| `cluster.py` | The distributed worker service and the coordinator merge |
| `cli.py` | Flags, argv handling, subcommands, and exit codes |

Dependencies point one way and never cycle. `errors` and `histogram` sit at the
bottom. `validation` sits above `errors`, and `config` uses both. `sources` and
`metrics` build on `config`. `engine`, `reporting`, and `diff` sit above those.
`html_report` builds on `reporting`, `cluster` builds on `engine`, and `cli` uses
everything. Keep it that way.

Keep the module count down. Three request sources became one `RequestCycle`. The
profile, banner, and Prometheus modules folded into their neighbors. The
Prometheus writers are now `render_prometheus` and `write_prometheus` in
`reporting`. Prefer a new function in an existing module over a new file.

### Conventions

**Validate at the edge.** `RunPlan.__post_init__` rejects a bad plan, so the
engine never guards against nonsense it cannot receive. New options get their
checks there.

**One exception type reaches the user.** Raise `TestBusterError` for anything a
person can fix, with a message that names the flag. Let every other exception
escape so the traceback survives.

```python
raise TestBusterError(f"--concurrency must be at least 1, got {workers}")
```

**Requests never raise.** Every failure inside `_single_shot` becomes an
`Attempt` with status `0` and a short reason. One dead host must not tear down
the worker pool. New failure modes get a branch in `classify_failure`.

**Group failures into short names.** A run against a dead host prints one line
with a count, not ten thousand tracebacks. Add a bucket rather than passing an
exception string through.

**Sort before rendering.** Every map that reaches a report gets sorted. Two runs
over the same data then print identical lines, and two reports diff cleanly.

**Watch the request path.** Anything held per request multiplies by the request
count. A source fills its templates at load time, latencies live in `array('d')`,
and bodies stream through a fixed buffer. A ten million request run must not grow.

**Guard the divisions.** Reports get built from runs with zero requests and zero
elapsed time. Every rate goes through an `if total else 0.0`.

### Comments and docstrings

Prose in this codebase follows ASD-STE100 Simplified Technical English:

- Active voice. "The parser reads the file", not "the file is read".
- No contractions, and no semicolons.
- Short common words: use, not utilize. Help, not facilitate. Make sure, not
  ensure.
- No marketing adjectives: no seamless, robust, powerful, or blazingly fast.
- One instruction per sentence, 20 words at most.
- One name per thing. Do not call the worker pool a task pool elsewhere.

Say what the code does, or why it exists. Skip what the function name already
says.

```python
# Yes
# aiohttp rejects keepalive_timeout together with force_close, so the two
# settings have to stay mutually exclusive.

# No
# This seamlessly ensures that the robust connection handling logic is
# properly leveraged.
```

### Tests

Eight files hold the suite, and each covers one module or one pair.
`test_reporting.py` covers the console, JSON, CSV, NDJSON, HTML, and Prometheus
output. Add to the file that owns the behavior rather than starting a new one.

`pytest-asyncio` runs in auto mode, so `async def test_*` needs no marker.

Engine tests run against a real aiohttp server on a loopback port, from the
`server` fixture in `tests/conftest.py`. A load generator lives or dies on real
socket behavior, so avoid mocking the session. Add a route to that fixture for a
behavior you need.

Name a test after the behavior it pins down, not the function it calls:

```python
def test_never_overshoots_the_request_cap(self): ...  # yes
def test_claim(self): ...  # no
```

When you fix a bug, leave a comment on the test saying what broke:

```python
def test_survives_a_zero_length_run(self):
    # A zero-length wall time must not divide by zero.
```

Timing tests need slack. CI runners are noisy. Assert a direction and a generous
bound rather than an exact number.

### Adding a flag

1. Add the field to `RunPlan`, with its check in `__post_init__`.
2. Add the key to `RunPlan.to_dict`, so it lands in the JSON report.
3. Declare the option in `cli.py` with the factory for its help panel. The
   factories are `_req`, `_load`, `_net`, `_retry`, `_check`, `_gate`, and
   `_out`. Keep the declaration to one line where the help text fits.
4. Thread it through `run_command`.
5. Add it to the README table for its group.
6. Test the validation and the behavior.

A flag that changes what the report contains needs a new `schema` string in
`Report.to_dict`. Say so in the pull request.

### Pull requests

- One concern per pull request.
- Say what changed and why. Include before and after numbers for a performance
  claim.
- Note anything that changes output shape or exit codes. Both are a contract.
- New behavior needs a test.
- A change that removes code and keeps the four checks green is welcome.

### Reporting a bug

Include:

- The full command, with secrets removed.
- `testbuster version` output.
- What you expected, and what happened.
- The JSON report from `--output` where you can share it.

## Built with

[aiohttp](https://docs.aiohttp.org/), [httpx](https://www.python-httpx.org/),
[Typer](https://typer.tiangolo.com/), and [Rich](https://rich.readthedocs.io/).
Inspired by wrk, hey, vegeta, and Apache Bench.
