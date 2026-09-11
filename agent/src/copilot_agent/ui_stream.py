"""The AI SDK UI message stream protocol, spoken by the Python service.

This is what `useChat` reads (DefaultChatTransport in ai 7.0.14): one JSON chunk per SSE event,
`data: <json>` then a blank line, and `data: [DONE]` at the end. The client parses every chunk
with uiMessageChunkSchema, a union of zod STRICT objects, and the first chunk that fails kills
the whole stream. Two things fail it: a key the schema does not list, and null in an optional
field ({"type": "start", "messageId": null} is rejected). So chunks here are plain dicts built
by the functions below, each writing exactly the keys it sets. There is no model to dump, so
there is no exclude_none to forget.

ui_message_chunks() adapts one run of the chat graph (graph.py) to that protocol. It produces
what app/api/chat/route.ts produces for the same run:

    canned:    start, data-retrieval, text-start, text-delta (whole reply), text-end,
               data-signature, finish
    answered:  start, data-retrieval, data-sources (if any), start-step, text-start,
               text-delta x N, text-end, finish-step, data-signature, finish
    failure:   an error chunk "Stream failed" (the route's onError text), then nothing

Two ordering rules carry weight, both checked in tests/python-stream-contract.test.ts against
the real Chat class:

  - `start` goes out WITH the first data part, not when the request arrives (invariant 12 has
    the history). The graph spends seconds planning and retrieving before there is anything to
    show. A `start` sent at once and followed by a failure leaves an EMPTY assistant bubble above
    the error, which is the Day 15 symptom by another road. TypeScript gets this for free: it
    returns a 500 before streaming. Here the status is already 200, so the error chunk alone,
    with no `start`, is what gives the reader the same screen: no bubble, the error box.
  - `start-step` and `text-start` go out with the first token, as the AI SDK sends them. A model
    call that fails before its first token leaves no empty step behind.

The signature goes out before `finish`. In TypeScript its place is a race between the merged
stream and `await result.text`; the client accepts it on either side, and here it is fixed.
"""

import json
import logging
import secrets
import string
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from typing import Any

from copilot_agent.generation import FinishReason
from copilot_agent.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# One UI message stream chunk, as JSON-ready data.
Chunk = dict[str, Any]

# Headers the AI SDK's createUIMessageStreamResponse adds on top of the SSE ones. FastAPI's
# EventSourceResponse already sets content-type, cache-control: no-cache and
# x-accel-buffering: no.
UI_MESSAGE_STREAM_HEADERS = {"x-vercel-ai-ui-message-stream": "v1"}

DONE = "[DONE]"

# The route's onError text. The reader never sees it (the UI shows its own error box), and the
# real reason stays in the server log: an exception message can quote anything, a key included.
STREAM_FAILED = "Stream failed"

# Text part ids. Any string unique within the message works; "canned" is the route's.
CANNED_TEXT_ID = "canned"
ANSWER_TEXT_ID = "answer"

# generateId() in ai: 16 characters from this alphabet.
_ID_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase
_ID_LENGTH = 16


def new_message_id() -> str:
    """A message id shaped like the AI SDK's (createUIMessageStream stamps one on `start`)."""
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_LENGTH))


def encode(chunk: Chunk | str) -> str:
    """One chunk as the text after `data: `. Compact like JSON.stringify, non-ASCII kept as is.

    allow_nan=False: Python would write NaN, which is not JSON, and the client's parse would kill
    the stream. Failing here puts the bug in the server log instead.
    """
    if isinstance(chunk, str):
        return chunk
    return json.dumps(chunk, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# ---- the chunks this service sends (a subset of uiMessageChunkSchema) ----------------------


def start(message_id: str) -> Chunk:
    return {"type": "start", "messageId": message_id}


def start_step() -> Chunk:
    return {"type": "start-step"}


def text_start(text_id: str) -> Chunk:
    return {"type": "text-start", "id": text_id}


def text_delta(text_id: str, delta: str) -> Chunk:
    return {"type": "text-delta", "id": text_id, "delta": delta}


def text_end(text_id: str) -> Chunk:
    return {"type": "text-end", "id": text_id}


def finish_step() -> Chunk:
    return {"type": "finish-step"}


def finish(finish_reason: FinishReason | None = None) -> Chunk:
    # The canned path sends a bare finish, like the route's writer.write({ type: "finish" }).
    if finish_reason is None:
        return {"type": "finish"}
    return {"type": "finish", "finishReason": finish_reason}


def data(name: str, value: object) -> Chunk:
    # The custom parts of lib/chat-types.ts (ChatDataParts): retrieval, sources, signature.
    return {"type": f"data-{name}", "data": value}


def error(error_text: str) -> Chunk:
    return {"type": "error", "errorText": error_text}


def source_pills(relevant: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """toSourcePills from lib/sources.ts: one pill per PAGE, carrying its chunk numbers.

    The prompt numbers chunks "[Source 1]" to "[Source N]" in retrieval order, and the model
    cites those numbers; a pill keeps the best score of its page and the numbers it stands for.
    The contract test compares the result with the TypeScript function's on the same chunks.
    """
    by_url: dict[str, dict[str, Any]] = {}
    for n, chunk in enumerate(relevant, start=1):
        pill = by_url.get(chunk.source_url)
        if pill is not None:
            pill["chunks"].append(n)
            pill["score"] = max(pill["score"], chunk.score)
            continue
        by_url[chunk.source_url] = {
            "id": len(by_url) + 1,
            "title": chunk.title,
            "url": chunk.source_url,
            "score": chunk.score,
            "chunks": [n],
        }
    return list(by_url.values())


# ---- the adapter ------------------------------------------------------------------------------


async def ui_message_chunks(
    parts: AsyncIterator[dict[str, Any]],
    *,
    message_id: str,
    sign: Callable[[str], str],
) -> AsyncIterator[Chunk]:
    """Turn one graph run into UI message chunks.

    `parts` is graph.astream(..., stream_mode=["updates", "messages"], version="v2"). `sign`
    signs an answer's text (signing.sign_assistant_text with the secret bound).

    An Exception from the graph becomes one error chunk and ends the stream; it is logged here,
    and never sent. Cancellation is not an Exception (CancelledError is a BaseException): when
    the client disconnects it passes straight through and cancels the run. aclosing() makes
    closing this generator close the graph's stream too, instead of leaving it to the garbage
    collector.
    """
    intent: str | None = None
    streamed: list[str] = []
    text_open = False

    async with aclosing(parts) as events:
        try:
            async for part in events:
                if part["type"] == "messages":
                    message, metadata = part["data"]
                    # Only the answer. The planner is tagged nostream (graph.py), so this is a
                    # second fence, not the first.
                    if metadata.get("langgraph_node") != "generate" or not message.text:
                        continue
                    if not text_open:
                        yield start_step()
                        yield text_start(ANSWER_TEXT_ID)
                        text_open = True
                    streamed.append(message.text)
                    yield text_delta(ANSWER_TEXT_ID, message.text)
                    continue

                for node, update in part["data"].items():
                    if node == "plan":
                        intent = update["plan"].intent
                    elif node == "canned":
                        reply = update["answer"]
                        yield start(message_id)
                        yield data("retrieval", {"mode": update["mode"], "intent": intent})
                        yield text_start(CANNED_TEXT_ID)
                        yield text_delta(CANNED_TEXT_ID, reply)
                        yield text_end(CANNED_TEXT_ID)
                        # Canned replies are assistant turns too: unsigned, they would be dropped
                        # from the next request's history.
                        yield data("signature", {"sig": sign(reply)})
                        yield finish()
                    elif node == "merge":
                        yield start(message_id)
                        yield data("retrieval", {"mode": update["mode"], "intent": intent})
                        if update["relevant"]:
                            yield data("sources", source_pills(update["relevant"]))
                    elif node == "generate":
                        if not text_open:
                            # An empty answer still gets its (empty) text part, as TypeScript's
                            # does: the model's output item opens it there.
                            yield start_step()
                            yield text_start(ANSWER_TEXT_ID)
                        yield text_end(ANSWER_TEXT_ID)
                        yield finish_step()
                        # Sign what the client received, joined, as the route signs result.text.
                        yield data("signature", {"sig": sign("".join(streamed))})
                        yield finish(update["generation"].finish_reason)
        except Exception:
            logger.exception("chat stream failed")
            yield error(STREAM_FAILED)
