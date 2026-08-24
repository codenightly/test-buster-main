"""Test Buster! - an async HTTP load generator.

The public surface is small on purpose. Import RunPlan and LoadEngine to drive
a load test from your own code, and render_report to print the result.
"""

from __future__ import annotations

from importlib import metadata

#: Display name. Use this in banners, reports, and user-facing text.
APP_NAME = "Test Buster!"

#: Executable name. A shell command cannot hold a space or an exclamation mark.
COMMAND_NAME = "testbuster"

try:
    __version__ = metadata.version("test-buster")
except metadata.PackageNotFoundError:  # a source checkout with no install
    __version__ = "0.0.0+dev"

__all__ = ["APP_NAME", "COMMAND_NAME", "__version__"]
