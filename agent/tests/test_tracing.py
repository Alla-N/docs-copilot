"""Tracing: the mask, the run config, and the promise that none of it can fail a request.

Offline. Nothing here builds a real Langfuse client or sends a span; what is checked is what the
service decides before anything is exported.
"""

from types import MappingProxyType
from typing import Any

import pytest
from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanData,
    OtelSpanIdentifier,
)

from copilot_agent.tracing import REDACTED, Tracing, redact_values

TRACE_ID = "0123456789abcdef0123456789abcdef"


def span_batch(attributes: dict[str, Any]) -> tuple[OtelSpanIdentifier, MaskOtelSpansParams]:
    """One span in one export batch, as Langfuse hands it to a mask."""
    identifier = OtelSpanIdentifier(trace_id=TRACE_ID, span_id="0123456789abcdef")
    span = OtelSpanData(
        trace_id=identifier.trace_id,
        span_id=identifier.span_id,
        parent_span_id=None,
        name="chat",
        instrumentation_scope_name="langfuse",
        instrumentation_scope_version=None,
        attributes=MappingProxyType(attributes),
        resource_attributes=MappingProxyType({}),
    )
    return identifier, MaskOtelSpansParams(spans=MappingProxyType({identifier: span}))


def patched(result: MaskOtelSpansResult | None, identifier: OtelSpanIdentifier) -> dict[str, Any]:
    assert result is not None
    return dict(result.span_patches[identifier].set_attributes)


# ---- the mask ------------------------------------------------------------------------------


def test_a_secret_inside_an_attribute_is_replaced() -> None:
    mask = redact_values(["sk-secret-key", "postgresql://user:pw@host/db"])
    identifier, params = span_batch(
        {
            "langfuse.observation.status_message": "401 from https://api.openai.com "
            "with Authorization: Bearer sk-secret-key",
            "langfuse.observation.input": "how do I stream text",
        }
    )

    changed = patched(mask(params=params), identifier)

    assert "sk-secret-key" not in changed["langfuse.observation.status_message"]
    assert REDACTED in changed["langfuse.observation.status_message"]
    # Only what changed is patched: an untouched attribute is not resent.
    assert "langfuse.observation.input" not in changed


def test_a_secret_inside_a_list_attribute_is_replaced() -> None:
    # OpenTelemetry attributes can be homogeneous sequences, and a patch must return one.
    mask = redact_values(["sk-secret-key"])
    identifier, params = span_batch({"langfuse.observation.metadata.args": ["a", "sk-secret-key"]})

    assert patched(mask(params=params), identifier) == {
        "langfuse.observation.metadata.args": ["a", REDACTED]
    }


def test_a_batch_that_holds_no_secret_is_left_alone() -> None:
    mask = redact_values(["sk-secret-key"])
    _, params = span_batch({"langfuse.observation.input": "how do I stream text"})

    assert mask(params=params) is None


def test_a_secret_that_contains_another_is_replaced_whole() -> None:
    # Longest first. The other order would leave the tail of the longer value in the span.
    mask = redact_values(["abcd", "abcd-efgh-the-real-key"])
    identifier, params = span_batch({"langfuse.observation.output": "key abcd-efgh-the-real-key"})

    assert patched(mask(params=params), identifier) == {
        "langfuse.observation.output": f"key {REDACTED}"
    }


def test_an_empty_secret_shreds_nothing() -> None:
    # settings.secret_values() skips unset secrets, but an empty string must not become a needle
    # that matches at every position.
    mask = redact_values(["", "   "])
    _, params = span_batch({"langfuse.observation.input": "how do I stream text"})

    assert mask(params=params) is None


def test_a_mask_that_cannot_read_a_span_exports_the_batch_unmasked() -> None:
    # Raising here costs the whole batch, traces with no secret in them included.
    class Exploding:
        @property
        def spans(self) -> dict[OtelSpanIdentifier, OtelSpanData]:
            raise RuntimeError("no")

    mask = redact_values(["sk-secret-key"])

    assert mask(params=Exploding()) is None


# ---- tracing that is off -------------------------------------------------------------------


def test_tracing_off_makes_no_trace_id_and_no_run_config() -> None:
    tracing = Tracing()

    assert tracing.enabled is False
    assert tracing.new_trace_id() is None
    assert tracing.run_config(trace_id=None, session_id="thread", tags=["web"]) == {}


def test_recording_an_error_with_tracing_off_does_nothing() -> None:
    Tracing().record_error(None, RuntimeError("boom"))


# ---- tracing that is on --------------------------------------------------------------------


class FakeClient:
    """Stands in for Langfuse: records what the service asked it to do."""

    def __init__(self, *, fail: bool = False) -> None:
        self.observations: list[dict[str, Any]] = []
        self.ended = 0
        self._fail = fail

    def start_observation(self, **kwargs: Any) -> "FakeClient":
        if self._fail:
            raise RuntimeError("exporter is unhappy")
        self.observations.append(kwargs)
        return self

    def end(self) -> None:
        self.ended += 1


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def test_the_run_config_carries_the_trace_id_and_the_session(client: FakeClient) -> None:
    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]

    config = tracing.run_config(trace_id=TRACE_ID, session_id="thread-abc", tags=["eval"])

    (handler,) = config["callbacks"]
    # The handler nests the whole run under an id we made ourselves, which is the id the
    # query_log row keeps. Private attribute on purpose: this is the only way to see it.
    assert handler._trace_context == {"trace_id": TRACE_ID}
    assert config["metadata"] == {
        "langfuse_session_id": "thread-abc",
        "langfuse_tags": ["eval"],
    }
    # A handler per run, never shared: it maps THIS run's LangChain run ids.
    other = tracing.run_config(trace_id=TRACE_ID, session_id="thread-abc")
    assert other["callbacks"][0] is not handler


def test_a_trace_id_is_32_hex_characters_and_new_every_time(client: FakeClient) -> None:
    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]

    first = tracing.new_trace_id()
    second = tracing.new_trace_id()

    assert first is not None and len(first) == 32
    assert set(first) <= set("0123456789abcdef")
    assert first != second


def test_an_error_becomes_an_error_span_on_the_same_trace(client: FakeClient) -> None:
    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]

    tracing.record_error(TRACE_ID, ValueError("no signature"))

    (observation,) = client.observations
    assert observation["trace_context"] == {"trace_id": TRACE_ID}
    assert observation["level"] == "ERROR"
    assert observation["status_message"] == "ValueError: no signature"
    assert client.ended == 1


def test_the_grounding_chunks_go_on_the_trace_as_json(client: FakeClient) -> None:
    from copilot_agent.retrieval import RetrievedChunk

    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]
    chunk = RetrievedChunk(
        content="Use streamText to stream a model response.",
        title="Streaming",
        source_url="https://ai-sdk.dev/docs/streaming",
        score=0.81,
    )

    tracing.record_context(TRACE_ID, [chunk])

    (observation,) = client.observations
    assert observation["trace_context"] == {"trace_id": TRACE_ID}
    assert observation["as_type"] == "retriever"
    # The texts, which the stream does not carry: this is what the judge reads back.
    assert observation["output"] == [
        {
            "title": "Streaming",
            "url": "https://ai-sdk.dev/docs/streaming",
            "score": 0.81,
            "text": "Use streamText to stream a model response.",
        }
    ]
    assert client.ended == 1


def test_an_answered_turn_that_kept_nothing_still_records_its_empty_context(
    client: FakeClient,
) -> None:
    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]

    tracing.record_context(TRACE_ID, [])

    assert client.observations[0]["output"] == []


def test_tracing_that_fails_never_reaches_the_caller() -> None:
    tracing = Tracing(client=FakeClient(fail=True), public_key="pk")  # type: ignore[arg-type]

    tracing.record_error(TRACE_ID, RuntimeError("boom"))
    tracing.record_context(TRACE_ID, [])


# ---- what the rest of the service does with it ----------------------------------------------


def test_a_traced_turn_puts_its_trace_id_on_the_query_log_row() -> None:
    from copilot_agent.query_log import Turn

    turn = Turn(question="q", thread_id="thread-abcdefghij123", origin="eval", trace_id=TRACE_ID)
    turn.answer = "an answer"
    turn.mode = "skipped"
    turn.retrieval_ms = 12.0

    assert turn.row()["trace_id"] == TRACE_ID


def test_an_untraced_turn_still_writes_its_row() -> None:
    from copilot_agent.query_log import Turn

    turn = Turn(question="q", thread_id="thread-abcdefghij123", origin="web")
    turn.answer = "an answer"
    turn.mode = "skipped"
    turn.retrieval_ms = 12.0

    assert turn.row()["trace_id"] is None


def chat_request(origin: str = "web") -> Any:
    from copilot_agent.api import ChatRequest

    return ChatRequest(
        thread_id="thread-abcdefghij123",
        question="how do I stream text",
        origin=origin,  # type: ignore[arg-type]
    )


def test_with_tracing_off_the_run_config_is_the_one_from_2_6() -> None:
    from copilot_agent.api import chat_run_config

    config = chat_run_config(chat_request(), Tracing(), None)

    assert config == {"configurable": {"thread_id": "thread-abcdefghij123"}}


def test_with_tracing_on_the_run_config_adds_the_handler_and_the_session(
    client: FakeClient,
) -> None:
    from copilot_agent.api import chat_run_config

    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]
    config = chat_run_config(chat_request("eval"), tracing, TRACE_ID)

    assert config["configurable"] == {"thread_id": "thread-abcdefghij123"}
    assert config["callbacks"][0]._trace_context == {"trace_id": TRACE_ID}
    # The origin is a tag, so an eval run can be filtered out of the dashboards the way the
    # query_log views filter it out of the traffic numbers.
    assert config["metadata"] == {
        "langfuse_session_id": "thread-abcdefghij123",
        "langfuse_tags": ["eval"],
    }


# ---- through the endpoint --------------------------------------------------------------------

POOLER_URL = "postgresql://u:not-real@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
AGENT_KEY = "k" * 64


class RecordingGraph:
    """A graph that runs nothing and keeps the config it was given."""

    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    def astream(self, state: Any, config: Any, **kwargs: Any) -> Any:
        self.configs.append(config)

        async def nothing() -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        return nothing()


@pytest.mark.anyio
async def test_the_endpoint_hands_the_graph_run_its_trace(client: FakeClient) -> None:
    """The wiring, not the pieces: a POST reaches the graph with the callbacks and the session.

    Reverting api.chat to the 2.6 config dict passes every other test in this file.
    """
    from contextlib import asynccontextmanager

    import httpx

    from copilot_agent.api import create_app
    from copilot_agent.query_log import QueryLog
    from copilot_agent.settings import Settings

    graph = RecordingGraph()
    tracing = Tracing(client=client, public_key="pk-lf-test")  # type: ignore[arg-type]

    @asynccontextmanager
    async def nothing_factory(settings: Settings) -> Any:
        yield None

    @asynccontextmanager
    async def tracing_factory(settings: Settings) -> Any:
        yield tracing

    async def never_inserts(row: Any) -> None:
        raise AssertionError("a run that produced nothing has no row")

    @asynccontextmanager
    async def query_log_factory(settings: Settings) -> Any:
        yield QueryLog(never_inserts)

    app = create_app(
        Settings(
            _env_file=None,
            openai_api_key="sk-test-not-real",
            cohere_api_key="co-test-not-real",
            database_url=POOLER_URL,
            agent_api_key=AGENT_KEY,
            assistant_signing_secret="test-signing-secret",
        ),
        search_factory=nothing_factory,
        checkpointer_factory=nothing_factory,
        graph_factory=lambda settings, search, checkpointer: graph,
        query_log_factory=query_log_factory,
        tracing_factory=tracing_factory,
    )

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as http,
    ):
        response = await http.post(
            "/chat",
            json={
                "thread_id": "thread-abcdefghij123",
                "question": "how do I stream text",
                "origin": "web",
            },
            headers={"Authorization": f"Bearer {AGENT_KEY}"},
        )

    assert response.status_code == 200
    (config,) = graph.configs
    assert config["configurable"] == {"thread_id": "thread-abcdefghij123"}
    assert config["metadata"]["langfuse_session_id"] == "thread-abcdefghij123"
    # The id the row will carry is the id the run was traced under.
    assert len(config["callbacks"][0]._trace_context["trace_id"]) == 32
