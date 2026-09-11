"""The conversation history the planner and the model read, from the thread's stored turns.

Since step 2.5 the history comes from the server, not from the client: the checkpointer keeps
each conversation's turns under its thread id (graph.py), and a /chat request carries only the
thread id and the new question. So there is no client-supplied assistant text left to forge or
to replay into another conversation, which is the gap the HMAC signatures (invariant 8) could
only narrow.

What the model reads stays what TypeScript gives it. lib/chat-request.ts (parseChatRequest)
caps the client's messages before anything reads them, and capped_history() is the same walk
over the stored turns:
  - the last MAX_MESSAGES messages, the new question included;
  - each text cut to MAX_CHARS_PER_MESSAGE characters, then trimmed; an empty one is skipped;
  - walking newest to oldest, stop at the first message that would take the total past
    MAX_TOTAL_CHARS, so the OLDEST turns are the ones dropped, never the question.
tests/test_history.py compares it with parseChatRequest itself, through a golden file written by
scripts/experiments/history-caps.ts.

One named difference: TypeScript counts and cuts in UTF-16 code units, Python in code points. An
astral character (an emoji) is two units and one code point, so on a text with one right at the
4000 cut, TypeScript keeps half of it (a lone surrogate) and Python keeps all of it; and Python
never counts more than TypeScript, so near the total cap it may keep a turn TypeScript drops.
"""

from collections.abc import Sequence

from copilot_agent.planner import HistoryTurn, js_trim

# lib/chat-request.ts; tests/test_ts_parity.py pins all three.
MAX_MESSAGES = 20
MAX_CHARS_PER_MESSAGE = 4000
MAX_TOTAL_CHARS = 24000

# The most one request can ever read next to its question. Storing more would be dead weight
# that every later checkpoint of the thread carries.
MAX_STORED_TURNS = MAX_MESSAGES - 1


def keep_recent(stored: list[HistoryTurn], new: list[HistoryTurn]) -> list[HistoryTurn]:
    """The reducer of the thread's `turns`: append, keep the last MAX_STORED_TURNS.

    Keeping only those changes nothing the model reads: capped_history() looks at no more.
    """
    return (stored + new)[-MAX_STORED_TURNS:]


def capped_history(turns: Sequence[HistoryTurn], question: str) -> list[HistoryTurn]:
    """The turns before `question` that parseChatRequest would keep, oldest first.

    `question` counts towards the total first, as the newest message. It arrives already within
    the per-message cap (api.ChatRequest), and the Next.js route forwards it already trimmed.
    """
    recent = list(turns)[-(MAX_MESSAGES - 1) :]
    total = len(question)
    kept: list[HistoryTurn] = []
    for turn in reversed(recent):
        text = js_trim(turn.text[:MAX_CHARS_PER_MESSAGE])
        if not text:
            continue
        if total + len(text) > MAX_TOTAL_CHARS:
            break
        total += len(text)
        kept.append(HistoryTurn(turn.role, text))
    kept.reverse()
    return kept
