"""Shared fixtures.

The engine tests run against a real aiohttp server on a loopback port rather
than a mocked session. A load generator lives or dies on real socket behavior,
so the tests exercise real sockets.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pytest
from aiohttp import web

from testbuster.metrics import Attempt


@dataclass(slots=True)
class StubServer:
    """A running test server and the counters its handlers update."""

    base_url: str
    hits: dict[str, int]

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


@pytest.fixture
async def server() -> AsyncIterator[StubServer]:
    """Start a small server with one route per behavior the engine must handle."""
    hits: dict[str, int] = {}

    def count(name: str) -> int:
        hits[name] = hits.get(name, 0) + 1
        return hits[name]

    async def ok(_request: web.Request) -> web.Response:
        count("ok")
        return web.Response(text="hello world", content_type="text/plain")

    async def echo(request: web.Request) -> web.Response:
        count("echo")
        body = await request.text()
        return web.json_response({"method": request.method, "body": body})

    async def boom(_request: web.Request) -> web.Response:
        count("boom")
        return web.Response(status=500, text="server on fire")

    async def flaky(_request: web.Request) -> web.Response:
        # Fail the first call, then succeed. This proves --retries works.
        if count("flaky") == 1:
            return web.Response(status=503, text="try again")
        return web.Response(status=200, text="recovered")

    async def slow(_request: web.Request) -> web.Response:
        count("slow")
        await asyncio.sleep(0.4)
        return web.Response(text="eventually")

    async def headers(request: web.Request) -> web.Response:
        count("headers")
        return web.json_response(dict(request.headers))

    async def payload(_request: web.Request) -> web.Response:
        count("payload")
        return web.Response(body=b"x" * 4096, content_type="application/octet-stream")

    app = web.Application()
    app.router.add_get("/ok", ok)
    app.router.add_route("*", "/echo", echo)
    app.router.add_get("/boom", boom)
    app.router.add_get("/flaky", flaky)
    app.router.add_get("/slow", slow)
    app.router.add_get("/headers", headers)
    app.router.add_get("/payload", payload)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sockets = runner.addresses
    host, port = sockets[0][0], sockets[0][1]

    try:
        yield StubServer(base_url=f"http://{host}:{port}", hits=hits)
    finally:
        await runner.cleanup()


@pytest.fixture
def make_attempt() -> Callable[..., Attempt]:
    """Build an Attempt with sensible defaults, overriding only what matters."""

    def build(
        status: int = 200,
        elapsed: float = 0.010,
        bytes_in: int = 100,
        failure: str | None = None,
        ttfb: float | None = 0.005,
        dns: float | None = None,
        connect: float | None = None,
        retries: int = 0,
    ) -> Attempt:
        return Attempt(
            status=status,
            elapsed=elapsed,
            bytes_in=bytes_in,
            failure=failure,
            ttfb=ttfb,
            dns=dns,
            connect=connect,
            retries=retries,
        )

    return build
