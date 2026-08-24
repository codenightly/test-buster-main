#!/usr/bin/env python3
"""Build a standalone Test Buster! executable, then prove it works.

Two backends:

  pyinstaller  Bundles CPython and every dependency into one self-extracting
               executable. Fast, needs no C compiler. This is the default.
  nuitka       Compiles the Python to C first, then builds a native binary.
               Starts faster and is harder to unpack. Needs a C compiler.

Neither produces a statically linked binary in the C sense: both embed the
CPython runtime rather than linking libpython into the image. What they do give
is the thing that matters here, one file that runs on a machine with no Python
installed.

Two details decide whether the result actually works, and both are handled here:

  1. The engine imports aiohttp_socks, httpx, uvloop, yaml, and aiohttp.web
     inside functions, most of them under try/except ImportError. A freezer's
     static analysis does not follow those, so they are passed explicitly. Only
     the ones importable in the current environment get bundled, so an extra you
     did not install cannot fail the build.
  2. testbuster reads its own version with importlib.metadata. A frozen app
     carries no dist-info unless it is asked to, so the binary would report
     0.0.0+dev. The metadata is copied in, and the smoke test fails the build if
     the version comes back wrong.

Usage:
    python freeze.py                     # one file, verified
    python freeze.py --backend nuitka    # true compilation, needs a compiler
    python freeze.py --onedir            # a folder, starts faster
    python freeze.py --tagged            # name it testbuster-linux-amd64
    python freeze.py --no-verify         # skip the smoke test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "src" / "testbuster" / "__main__.py"
DIST = ROOT / "dist"
WORK = ROOT / "build" / "freeze"

#: Distribution name, for the metadata the version lookup needs.
DIST_NAME = "test-buster"

#: Modules imported inside function bodies, which a freezer will not find.
#: Each entry is (module, why). Only importable ones are bundled.
LAZY_IMPORTS: tuple[tuple[str, str], ...] = (
    ("aiohttp.web", "cluster.build_worker_app builds the worker service"),
    ("aiohttp_socks", "the SOCKS connector, behind the socks extra"),
    ("httpx", "the HTTP/2 backend, behind the http2 extra"),
    ("h2", "httpx needs it to speak HTTP/2"),
    ("yaml", "YAML scenario files, behind the yaml extra"),
    ("uvloop", "the faster event loop, behind the speed extra"),
    ("typer._completion_shared", "the completion script generator"),
)

#: Never needed at runtime. Excluding them keeps the binary smaller.
EXCLUDES: tuple[str, ...] = ("tkinter", "_tkinter", "turtle", "pydoc_data", "lib2to3")


def say(message: str) -> None:
    print(f"[freeze] {message}", flush=True)


def die(message: str) -> None:
    print(f"[freeze] error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def available(module: str) -> bool:
    """Report whether a module can be imported in this environment."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_version() -> str:
    """Return the version testbuster reports when run from source."""
    from importlib import metadata

    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        die(
            f"{DIST_NAME} is not installed in this environment. "
            f'Run: python -m pip install -e "{ROOT}"'
        )
    raise AssertionError("unreachable")


def artifact_name(base: str, *, tagged: bool) -> str:
    """Return the output filename, optionally carrying platform and machine."""
    if tagged:
        machine = platform.machine().lower()
        machine = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64"}.get(machine, machine)
        base = f"{base}-{platform.system().lower()}-{machine}"
    return base + (".exe" if sys.platform == "win32" else "")


def ensure_backend(backend: str) -> None:
    """Check the build tool is importable, and say how to get it if not."""
    module = "PyInstaller" if backend == "pyinstaller" else "nuitka"
    if available(module):
        return
    extra = "freeze" if backend == "pyinstaller" else "nuitka"
    die(f'{backend} is not installed. Run: python -m pip install "{ROOT}[{extra}]"')


def pyinstaller_command(name: str, *, onefile: bool, lazy: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--console",
        "--onefile" if onefile else "--onedir",
        "--name",
        Path(name).stem,
        "--distpath",
        str(DIST),
        "--workpath",
        str(WORK),
        "--specpath",
        str(WORK),
        # Without this the frozen app has no dist-info and reports 0.0.0+dev.
        "--copy-metadata",
        DIST_NAME,
    ]
    for module in lazy:
        command += ["--hidden-import", module]
    for module in EXCLUDES:
        command += ["--exclude-module", module]
    command.append(str(ENTRY))
    return command


def nuitka_command(name: str, *, onefile: bool, lazy: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        f"--output-dir={DIST}",
        f"--output-filename={name}",
        "--include-package=testbuster",
        f"--include-distribution-metadata={DIST_NAME}",
    ]
    if onefile:
        command.append("--onefile")
    for module in lazy:
        # Nuitka wants a package flag for a dotted name it should follow whole.
        flag = "--include-package" if "." not in module else "--include-module"
        command.append(f"{flag}={module}")
    for module in EXCLUDES:
        command.append(f"--nofollow-import-to={module}")
    command.append(str(ENTRY))
    return command


def locate(name: str, *, onefile: bool, backend: str) -> Path:
    """Find the built executable, whichever layout the backend produced."""
    stem = Path(name).stem
    candidates = [DIST / name, DIST / stem / name]
    if backend == "nuitka":
        candidates += [DIST / f"{ENTRY.stem}.dist" / name, DIST / "__main__.dist" / name]
    for path in candidates:
        if path.is_file():
            return path
    found = sorted(p for p in DIST.rglob(stem + "*") if p.is_file() and p.suffix in {"", ".exe"})
    if found:
        return found[0]
    die(f"the build finished but no executable turned up under {DIST}")
    raise AssertionError("unreachable")


class _Handler(BaseHTTPRequestHandler):
    """Answers the smoke test's load run. Silent, so it does not spam stdout."""

    def do_GET(self) -> None:
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def run(binary: Path, *args: str, expect: int = 0) -> str:
    """Run the built binary and check its exit code."""
    done = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=180, check=False
    )
    if done.returncode != expect:
        sys.stderr.write(done.stdout + done.stderr)
        die(f"`{binary.name} {' '.join(args)}` exited {done.returncode}, wanted {expect}")
    return done.stdout


def verify(binary: Path, want_version: str) -> None:
    """Exercise the built binary against a real server."""
    say("verifying the binary")

    reported = run(binary, "version", "--plain").strip()
    if reported != want_version:
        die(
            f"the binary reports version {reported!r} but this source tree is "
            f"{want_version!r}. The distribution metadata did not make it in."
        )
    say(f"  version           {reported}")

    run(binary, "--help")
    run(binary, "run", "--help")
    say("  help             ok")

    for shell in ("bash", "zsh", "fish", "powershell"):
        if not run(binary, "completion", shell).strip():
            die(f"the {shell} completion script came back empty")
    say("  completion       ok (4 shells)")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/ok"
        raw = run(binary, url, "-n", "100", "-c", "10", "--quiet", "--json")
        report = json.loads(raw)
        total = report["summary"]["total_requests"]
        ok = report["summary"]["successful"]
        if (total, ok) != (100, 100):
            die(f"the load run reported {ok}/{total} successful, wanted 100/100")
        say(f"  load test        {ok}/{total} ok, p95 {report['latency']['p95']:.2f} ms")

        # A gate that must fail proves the exit-code contract survived freezing.
        run(binary, url, "-n", "20", "--fail-over-p95", "0.0001", "--quiet", expect=4)
        run(binary, "http://127.0.0.1:1/", "-n", "2", "-t", "1s", "--quiet", expect=3)
        run(binary, url, "-X", "PSOT", "--quiet", expect=2)
        say("  exit codes       ok (0, 2, 3, 4)")
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a standalone Test Buster! executable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("pyinstaller", "nuitka"),
        default="pyinstaller",
        help="pyinstaller bundles the runtime (default). nuitka compiles to C first.",
    )
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument(
        "--onefile", action="store_true", default=True, help="One executable (default)."
    )
    layout.add_argument(
        "--onedir",
        dest="onefile",
        action="store_false",
        help="A folder beside the executable. Starts faster, ships as an archive.",
    )
    parser.add_argument("--name", default="testbuster", help="Base output name.")
    parser.add_argument(
        "--tagged",
        action="store_true",
        help="Append the platform and machine, for release assets.",
    )
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip the checks.")
    parser.add_argument(
        "--keep-work", action="store_true", help="Leave the intermediate build tree in place."
    )
    args = parser.parse_args()

    if not ENTRY.is_file():
        die(f"cannot find the entry point at {ENTRY}. Run this from the repository root.")
    ensure_backend(args.backend)

    version = installed_version()
    name = artifact_name(args.name, tagged=args.tagged)

    lazy = [module for module, _ in LAZY_IMPORTS if available(module)]
    skipped = [module for module, _ in LAZY_IMPORTS if not available(module)]

    say(f"{args.backend} -> {name} ({'onefile' if args.onefile else 'onedir'}), version {version}")
    say(f"bundling optional modules: {', '.join(lazy) or 'none'}")
    if skipped:
        say(f"not installed, so left out: {', '.join(skipped)}")
        say('  install every extra first for a complete binary: pip install ".[all]"')

    shutil.rmtree(WORK, ignore_errors=True)
    builder = pyinstaller_command if args.backend == "pyinstaller" else nuitka_command
    command = builder(name, onefile=args.onefile, lazy=lazy)

    say("building, which takes a minute")
    done = subprocess.run(command, cwd=ROOT, check=False)
    if done.returncode != 0:
        die(f"{args.backend} exited {done.returncode}")

    binary = locate(name, onefile=args.onefile, backend=args.backend)
    if sys.platform != "win32":
        binary.chmod(0o755)

    if args.verify:
        verify(binary, version)

    if not args.keep_work:
        shutil.rmtree(WORK, ignore_errors=True)
        shutil.rmtree(ROOT / "build", ignore_errors=True)

    size = binary.stat().st_size / (1024 * 1024)
    say(f"done: {binary}  ({size:.1f} MiB)")
    if args.onefile:
        say("this one file needs no Python on the target machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
