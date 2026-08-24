"""Exit codes and the one exception type the CLI knows how to print."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes. Build scripts branch on these values."""

    OK = 0
    BAD_USAGE = 2  # the flags or the URL are wrong
    RUN_FAILED = 3  # the run could not start, or every request failed
    GATE_FAILED = 4  # the run finished but missed a --fail-* threshold
    INTERRUPTED = 130  # Ctrl+C, matching the shell convention


class TestBusterError(Exception):
    """A problem the user can fix.

    pytest tries to collect any class whose name starts with Test, so
    __test__ = False below keeps this exception out of test discovery.

    The CLI catches this, prints the message without a traceback, and exits
    with the attached code. Raise it for bad input and unmet preconditions.
    Let every other exception escape so the traceback survives.
    """

    __test__ = False

    def __init__(self, message: str, code: ExitCode = ExitCode.BAD_USAGE) -> None:
        super().__init__(message)
        self.code = code
