"""Langfuse tracing: one trace per /chat turn, or nothing at all (step 2.7).

Why the service needs it. The query log (query_log.py) records what a turn cost and how long it
took; the thread records what was said. Neither says WHY: which sub-queries the planner wrote,
what came back for each of them, what the model was actually shown. And one failure mode is
invisible everywhere else: an error inside a streamed answer travels as an error chunk in a 200
response (ui_stream.py), so the route sees a fine response, a load balancer counts no 5xx, and
the only trace of a paid, failed turn is a line in this process's log. A trace carries all of it.

What a trace is here. Langfuse 4.x IS OpenTelemetry: a trace is a tree of spans, and "generation",
"retriever" and the rest are span types with extra attributes. The LangChain callback handler
turns one graph run into that tree by itself: a span per node, a generation per model call with
its model, tokens, cost and time to first token. This module only has to do the three things the
handler cannot know about:

  - give the turn an id we keep. The trace id is made HERE, before the run starts, and goes into
    the query_log row (db/007_trace_id.sql), so every logged turn points at its trace, in
    production as in an eval run. The alternative, reading the id back out of a span, would tie
    the log to the tracing being on.
  - group the turns of one conversation. Langfuse sessions are the thread id, which is what
    `langfuse_session_id` in the run's metadata sets (CallbackHandler._parse_langfuse_trace_
    attributes reads it at the root of the chain).
  - record a failure that happens after the graph is done (signing, encoding): api.chat hands it
    to record_error, which hangs one ERROR span off the same trace id.

Nothing here creates a "current" span or enters a context manager that has to be exited later.
That is deliberate. The request path is an async generator consumed by an SSE producer task, and
a disconnect closes it from elsewhere; a contextvars token entered on one task and reset on
another raises, which is the same class of bug step 2.4 spent a day on. Passing
`trace_context={"trace_id": ...}` to the handler nests the whole run under our id with no context
to unwind.

Off by default, and never fatal. No keys, no tracing, and the service behaves exactly as it did
in 2.6 (the Langfuse client itself falls back to a no-op tracer when a key is missing, but this
module does not even build one). With keys, an export failure is the exporter's problem on its
own thread: it cannot reach the request. Observability that can take the service down is worse
than none.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse

# langfuse.langchain imports the `langchain` meta-package at module load and branches on its
# version; langchain-core is not enough, and the error only says "pip install langchain". Hence
# langchain in pyproject.toml, for this import alone (found by the test gate, 2.7).
from langfuse.langchain import CallbackHandler
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

from copilot_agent.retrieval import RetrievedChunk
from copilot_agent.settings import Settings

logger = logging.getLogger(__name__)

REDACTED = "[redacted]"

# The span that records a failure the graph never saw (see record_error).
ERROR_SPAN_NAME = "stream-failed"

# The span that carries the chunks the answer was grounded on (see record_context).
CONTEXT_SPAN_NAME = "context"

# How long shutdown waits for the exporter to send what is queued. It runs in a worker thread,
# so this is time the event loop is not blocked, but ECS still stops a task 30 s after asking.
SHUTDOWN_TIMEOUT_S = 5.0

MaskFunction = Callable[..., MaskOtelSpansResult | None]


def redact_values(secrets: Sequence[str]) -> MaskFunction:
    """A mask that removes KNOWN secret values from span attributes before they are exported.

    Not a guess at what a secret looks like: the caller passes the values this process holds (the
    OpenAI and Cohere keys, the database URL, the service key, the signing secret), and any
    attribute that contains one of them leaves without it. That covers the attributes this service
    never writes but the handler might: a provider error message quoting a request header, a
    connection string in an exception, a status_message from record_error.

    Langfuse calls this at export stage, on the batch processor's worker thread, with one batch of
    spans: `params.spans` maps (trace_id, span_id) to a read-only snapshot, and the result is a
    sparse patch per span. A hook that raises costs the WHOLE batch, so this one cannot raise.
    """
    # Longest first: one secret that contains another must be replaced whole. Blanks dropped, or
    # every attribute would be shredded by an empty needle.
    values = sorted({secret for secret in secrets if secret}, key=len, reverse=True)

    def redacted(value: Any) -> Any | None:
        """The value with every secret replaced, or None when it holds none."""
        if isinstance(value, str):
            masked = value
            for secret in values:
                if secret in masked:
                    masked = masked.replace(secret, REDACTED)
            return None if masked == value else masked
        if isinstance(value, (list, tuple)):
            items = []
            changed = False
            for item in value:
                masked_item = redacted(item)
                changed = changed or masked_item is not None
                items.append(item if masked_item is None else masked_item)
            return items if changed else None
        return None

    def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
        if not values:
            return None
        try:
            patches = {}
            for identifier, span in params.spans.items():
                changed = {
                    key: masked
                    for key, value in span.attributes.items()
                    if (masked := redacted(value)) is not None
                }
                if changed:
                    patches[identifier] = OtelSpanPatch(set_attributes=changed)
            return MaskOtelSpansResult(span_patches=patches) if patches else None
        except Exception:
            # Dropping the batch would lose traces that hold no secret at all.
            logger.exception("tracing: the mask failed; exporting this batch unmasked")
            return None

    return mask_otel_spans


@dataclass(frozen=True)
class Tracing:
    """The service's tracing, or the same object doing nothing.

    Built with no client when the keys are unset, so every call site is the same line whether
    tracing is on or off: no `if settings.langfuse_...` in api.py, and the tests get the disabled
    one for free.
    """

    client: Langfuse | None = None
    public_key: str | None = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def new_trace_id(self) -> str | None:
        """The id this turn's trace will have, made before the run starts. None when off.

        32 lowercase hex characters (W3C), which is why it is not the thread id: that one is the
        AI SDK's 16-character chat id. The two meet on the query_log row.
        """
        return Langfuse.create_trace_id() if self.client is not None else None

    def run_config(
        self, *, trace_id: str | None, session_id: str, tags: Sequence[str] = ()
    ) -> dict[str, Any]:
        """What to merge into the graph run's config so the run is traced. Empty when off.

        A fresh handler per run: it holds this run's trace context and its map of LangChain run
        ids. `metadata` is read by the handler at the root of the chain and becomes trace-level
        attributes; LangGraph also stores the run's metadata with the checkpoint, so these two
        keys (the thread id, which the checkpoint row already keys on, and the origin) are the
        only ones put there.
        """
        if self.client is None or trace_id is None:
            return {}
        return {
            "callbacks": [
                CallbackHandler(public_key=self.public_key, trace_context={"trace_id": trace_id})
            ],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_tags": list(tags),
            },
        }

    def record_context(self, trace_id: str | None, chunks: Sequence[RetrievedChunk]) -> None:
        """Put the chunks the model was shown on this turn's trace, as JSON.

        The stream carries PAGES (url, best score, chunk numbers), not texts: enough for recall,
        not enough to ask whether an answer is supported by what it was given. The handler's own
        retrieve spans hold each sub-query's result; this one holds the merged set the prompt was
        built from, which is what the faithfulness judge reads back (evals/agent-target.ts) and
        the first thing to look at when an answer is wrong.

        An answered turn with an EMPTY list is worth recording: it means nothing cleared the
        threshold and the model was told so. A canned turn retrieved nothing at all, and the
        caller skips it.
        """
        if self.client is None or trace_id is None:
            return
        try:
            self.client.start_observation(
                trace_context={"trace_id": trace_id},
                name=CONTEXT_SPAN_NAME,
                as_type="retriever",
                output=[
                    {
                        "title": chunk.title,
                        "url": chunk.source_url,
                        "score": chunk.score,
                        "text": chunk.content,
                    }
                    for chunk in chunks
                ],
            ).end()
        except Exception:
            logger.exception("tracing: could not record the grounding chunks")

    def record_error(self, trace_id: str | None, exc: BaseException) -> None:
        """Hang one ERROR span off this turn's trace, for a failure the graph did not raise.

        ui_stream turns any exception into one error chunk inside a 200 response and logs it
        here; without this the trace would end looking like a turn that simply produced no
        answer. The message is the exception's, which is why the mask exists: an exception can
        quote anything it was given.
        """
        if self.client is None or trace_id is None:
            return
        try:
            self.client.start_observation(
                trace_context={"trace_id": trace_id},
                name=ERROR_SPAN_NAME,
                as_type="span",
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
            ).end()
        except Exception:
            logger.exception("tracing: could not record the stream failure")


@asynccontextmanager
async def open_tracing(settings: Settings) -> AsyncIterator[Tracing]:
    """Tracing for the service's lifespan: a client when both keys are set, nothing otherwise.

    Opened FIRST in the lifespan so it closes LAST: the spans of the final requests, and of the
    query log's own shutdown, are still flushed after everything else has closed.
    """
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if not public_key or secret_key is None:
        logger.info("tracing off: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set")
        yield Tracing()
        return

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        mask_otel_spans=redact_values(settings.secret_values()),
    )
    try:
        yield Tracing(client=client, public_key=public_key)
    finally:
        # shutdown() flushes what is queued and stops the exporter threads. It BLOCKS, so it goes
        # to a thread: at this point the event loop is still running the lifespan, and a blocked
        # loop during a deploy means every in-flight response stalls with it.
        try:
            await asyncio.wait_for(asyncio.to_thread(client.shutdown), timeout=SHUTDOWN_TIMEOUT_S)
        except TimeoutError:
            logger.error("tracing: the exporter did not finish within %ss", SHUTDOWN_TIMEOUT_S)
        except Exception:
            logger.exception("tracing: shutdown failed")
