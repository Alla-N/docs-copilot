"""The refusal sentence and its detector: the Python port of lib/refusal.ts.

The query log (query_log.py) records, per request, whether the answer REFUSED. The eval set is
mined from that column (the suspicious_refusals view, db/002_query_log.sql), so the service must
call an answer a refusal exactly when the TypeScript route would. is_refusal() is a port of
isRefusal() and tests/test_refusal.py compares the two verdict by verdict, through a golden file
written by the TypeScript function itself (scripts/experiments/refusal-verdicts.ts).

lib/refusal.ts explains the rule (positional, then compositional) and the bugs that shaped it.
What this file adds is the porting traps, each one a place where the same regex text means
something else in Python:

  - JavaScript's `$` without the m flag is the end of the input; Python's `$` also matches just
    before a trailing newline. Here: `\\Z`, the literal translation. (No test can tell the two
    apart in this function: the answer is trimmed before the pattern sees it. A mutation run
    showed exactly that, so the golden has no case claiming otherwise.)
  - JavaScript's `\\s` and trim() use one whitespace set; Python's `\\s` and strip() use another
    (U+FEFF is whitespace only in JavaScript, \\x1c to \\x1f only in Python). Here: the JavaScript
    set, spelled out (planner.JS_WHITESPACE).
  - A JavaScript regex without the u flag has ASCII `\\b`, and its /i does not fold a non-ASCII
    letter into an ASCII one. Python's str patterns are Unicode by default: `\\b` sees accented
    letters as word characters, and re.IGNORECASE matches the long s as "s" (and the Kelvin
    sign as "k"). Here: re.ASCII.
"""

import re

from copilot_agent.planner import JS_WHITESPACE, js_trim

# lib/refusal.ts, verbatim; tests/test_ts_parity.py pins it. The prompt quotes it
# (generation.py), and the canned off-topic reply is it (graph.py).
REFUSAL_MESSAGE = (
    "I don't have information about that in the documentation. "
    "I can help with AI SDK docs. Ask me about those and I'll help."
)

# JavaScript's \s, as a character class body.
_WS = re.escape(JS_WHITESPACE)

# The load-bearing first sentence of REFUSAL_MESSAGE. Derived, as in TypeScript.
REFUSAL_CORE = REFUSAL_MESSAGE.split(". ")[0]

# Terminal punctuation followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(rf"[.!?][{_WS}]+")

# "The documentation doesn't cover <topic>." as an opening sentence (lib/refusal.ts has why the
# verb list is closed and why the topic may not hold ",;:"). Used with .match(), which anchors
# at the start like the JavaScript ^.
_NEGATIVE_OPENER = re.compile(
    r"the documentation (?:does(?:n't|n\u2019t| not) "
    r"(?:cover|mention|include|discuss|describe|contain|provide|address|explain)"
    r"|has no (?:information|details?|guidance))"
    rf"\b[^.,;:]*(?:[.!][{_WS}]*|\Z)",
    re.IGNORECASE | re.ASCII,
)


def _norm(text: str) -> str:
    # Lowercase, keep only [a-z0-9 ], collapse spaces. After the second step only spaces are
    # left to collapse, so JavaScript's \s+ and trim() act on spaces alone here.
    kept = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return re.sub(" +", " ", kept).strip(" ")


_CANONICAL_SENTENCES = {
    s for s in (_norm(part) for part in _SENTENCE_BOUNDARY.split(REFUSAL_MESSAGE)) if s
}


def is_refusal(answer: str) -> bool:
    """Does this answer REFUSE, not merely contain a refusal somewhere (lib/refusal.ts)."""
    raw = js_trim(answer)
    n = _norm(raw)
    core = _norm(REFUSAL_CORE)
    if n.startswith(core) or (n.startswith("the documentation") and core in n):
        return True

    opener = _NEGATIVE_OPENER.match(raw)
    if opener is None:
        return False
    rest = js_trim(raw[opener.end() :])
    if rest == "":
        return True
    sentences = (_norm(s) for s in _SENTENCE_BOUNDARY.split(rest))
    return all(s in _CANONICAL_SENTENCES for s in sentences if s)
