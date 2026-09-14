"""The shutdown budget: everything a stopping container waits for has to fit in ECS's 30 s.

ECS asks a task to stop with SIGTERM and kills it `stopTimeout` seconds later, 30 by default.
Everything that has to happen in between is a timeout somewhere in this service, and until this
file existed the numbers lived in four separate comments that never added up in one place.

The order is uvicorn's, from Server.shutdown() in uvicorn/server.py:

    close the listening sockets
    connection.shutdown() on every live connection
    await wait_for(_wait_tasks_to_complete(), timeout=config.timeout_graceful_shutdown)
    await lifespan.shutdown()          <- drain, the pool closes and the Langfuse flush are HERE

Two things in that order matter more than any single number. The lifespan runs LAST, after the
wait for in-flight connections, so anything that delays a connection delays the query log's
drain and the trace flush behind it. And `timeout_graceful_shutdown` defaults to None, which
means wait forever: an in-flight /chat stream would hold the shutdown open until ECS sent
SIGKILL, and the lifespan would then never run at all - the last query_log rows dropped, the
queued spans dropped, and three pools severed instead of closed, so Supavisor holds those server
connections until its own timeout. The Dockerfile passes the flag for exactly that reason, and
the test below is what keeps it passed.
"""

import json
import re
from pathlib import Path

import pytest

from copilot_agent.query_log import DRAIN_TIMEOUT_S
from copilot_agent.retrieval import POOL_CLOSE_TIMEOUT_S
from copilot_agent.tracing import SHUTDOWN_TIMEOUT_S

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"

# ECS's default stopTimeout, and the number the whole budget is sized against. Raising it is a
# task-definition change; assume the default, because Express Mode may not expose it.
ECS_STOP_TIMEOUT_S = 30.0

# The lifespan closes three pools, one after another, as its context managers unwind: the query
# log's, the checkpointer's and the search pool (api.create_app). Each waits POOL_CLOSE_TIMEOUT_S.
POOLS_CLOSED_AT_SHUTDOWN = 3

# Enough slack that a slow pool close or a second of scheduling noise is not a SIGKILL.
MIN_HEADROOM_S = 3.0


def dockerfile_text() -> str:
    """The Dockerfile with its backslash line continuations joined, as docker reads it."""
    return re.sub(r"\\\n\s*", " ", DOCKERFILE.read_text(encoding="utf-8"))


def dockerfile_cmd() -> list[str]:
    """The runtime CMD as a list of arguments (exec form, so it is JSON)."""
    for line in dockerfile_text().splitlines():
        if line.startswith("CMD "):
            return json.loads(line.removeprefix("CMD "))
    raise AssertionError(f"no exec-form CMD line in {DOCKERFILE}")


def cmd_option(name: str) -> str:
    cmd = dockerfile_cmd()
    assert name in cmd, f"{name} is not in the container's CMD: {cmd}"
    return cmd[cmd.index(name) + 1]


def dockerfile_env(name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", dockerfile_text(), re.M)
    assert match is not None, f"{name} is not set in {DOCKERFILE}"
    return match.group(1)


def test_the_container_bounds_the_wait_for_in_flight_streams() -> None:
    """Without --timeout-graceful-shutdown the lifespan never runs on a stop with a live stream.

    The value has to leave room for a normal answer to finish: the measured answered path reaches
    `done` at about 5.2 s median (agent/README.md, phase 2 done-when).
    """
    graceful = float(cmd_option("--timeout-graceful-shutdown"))
    assert graceful >= 6.0, "shorter than a normal answer: a stop would cancel turns in progress"


def test_the_worst_case_shutdown_fits_the_ecs_stop_timeout() -> None:
    graceful = float(cmd_option("--timeout-graceful-shutdown"))
    budget = (
        graceful
        + DRAIN_TIMEOUT_S
        + POOLS_CLOSED_AT_SHUTDOWN * POOL_CLOSE_TIMEOUT_S
        + SHUTDOWN_TIMEOUT_S
    )
    assert budget + MIN_HEADROOM_S <= ECS_STOP_TIMEOUT_S, (
        f"worst-case shutdown is {budget}s of a {ECS_STOP_TIMEOUT_S}s stop timeout: "
        f"streams {graceful} + drain {DRAIN_TIMEOUT_S} + "
        f"{POOLS_CLOSED_AT_SHUTDOWN} pools x {POOL_CLOSE_TIMEOUT_S} + flush {SHUTDOWN_TIMEOUT_S}"
    )


def test_the_exporter_thread_stops_before_our_own_cap() -> None:
    """tracing.py bounds the AWAIT on the flush, not the thread doing it.

    open_tracing runs client.shutdown() through asyncio.to_thread under a SHUTDOWN_TIMEOUT_S
    wait_for. When that fires, the coroutine stops waiting and the worker thread carries on: the
    OpenTelemetry BatchSpanProcessor underneath has its own export timeout, and its default is
    30000 ms - longer than our cap and longer than the whole ECS window, so the thread would
    outlive the process's chance to exit cleanly and the container would leave by SIGKILL. The
    Dockerfile sets OTEL_BSP_EXPORT_TIMEOUT below our cap so the thread's own bound is the
    tighter one (opentelemetry/sdk/trace/export/__init__.py reads the variable).
    """
    export_timeout_s = float(dockerfile_env("OTEL_BSP_EXPORT_TIMEOUT")) / 1000
    assert export_timeout_s < SHUTDOWN_TIMEOUT_S


def test_uvicorn_still_waits_forever_by_default() -> None:
    """A canary on the reason the flag exists, not on our own code.

    If uvicorn ever ships a bounded default, this fails and the Dockerfile flag becomes a choice
    rather than a fix. Same shape as the LangGraph cancellation canary in test_chat_api.py.
    """
    uvicorn = pytest.importorskip("uvicorn")
    assert uvicorn.Config("x").timeout_graceful_shutdown is None
