"""Request parity: the Python planner sends the TypeScript planner's request, plus one paragraph.

Step 3.5b is why "plus one paragraph" is in that sentence. The Python planner needs the vercel/ai
repository in scope and the TypeScript one does not, so the prompts had to differ. The two
available moves were both bad: delete this pin and the drift detector dies to make one commit
green, or fork the prompts and let them drift in silence. The third move is to express the
intended delta IN the pin, which is what planner.with_repository_scope() and
test_the_python_prompt_is_the_typescript_prompt_plus_one_paragraph do. Every byte outside that one
paragraph still has to match the golden, on both sides, so this file is a stricter oracle
afterwards than it was before: it now pins the delta as well as the base.

scripts/experiments/planner-requests.ts (npm run exp:planner-requests, from the repo root) ran
planQuery() on every planner-eval case with fetch replaced by a recorder, and froze each request
in tests/golden/planner-requests.json with the canned reply it got back. This test runs
plan_query() on the same cases through the real ChatOpenAI, with an httpx MockTransport in place
of the network, sends the same canned reply, and compares.

Free and deterministic, so it runs in the pre-commit hook and in CI. It is the exact half of the
planner port's parity. The statistical half is evals/planner_eval.py (real model, 5 runs a case):
once the requests match, a difference between the two planners' plans is the model, not the port.

The golden records a sha256 of lib/plan.ts and evals/planner-cases.ts. Change either and this
fails until the golden is regenerated, so the Python side cannot drift from an oracle that moved.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx2  # openai>=3 is built on httpx2, so the client handed to it must be an httpx2 one
import pytest

from copilot_agent.planner import (
    PLANNER_MAX_RETRIES,
    REPOSITORY_SCOPE_PARAGRAPH,
    SYSTEM_PROMPT,
    TS_SYSTEM_PROMPT,
    HistoryTurn,
    SubQuery,
    TokenUsage,
    build_planner,
    openai_planner_model,
    plan_query,
    with_repository_scope,
)
from copilot_agent.settings import Settings

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FILE = Path(__file__).parent / "golden" / "planner-requests.json"
GOLDEN: dict[str, Any] = json.loads(GOLDEN_FILE.read_text())
CASES = GOLDEN["cases"]
REGENERATE = "regenerate it from the repo root: npm run exp:planner-requests"


def encoded_by_langchain(ts_body: dict[str, Any]) -> dict[str, Any]:
    """The TypeScript body as LangChain encodes the same request, with step 3.5b's paragraph.

    Two encoding differences are expected, and nothing else is tolerated. Both are encodings of
    the same request, not a different one:
      - "stream": false. ChatOpenAI always sends its streaming flag; the AI SDK omits it, and
        false is the API's default.
      - "type": "message" on every input item. LangChain writes the item type out; the AI SDK
        leaves it implicit, and "message" is the only type an item with a role can have.
    If LangChain ever stops sending one of them, this fails and the list gets shorter.

    Then one deliberate difference, which is the whole of step 3.5b: the system prompt gains
    REPOSITORY_SCOPE_PARAGRAPH. It is spliced here by the same function the planner splices with,
    so this test cannot agree with a bug in the splice — a wrong anchor or a duplicated paragraph
    raises or shows up as a mismatch, rather than being mirrored into the expectation. Everything
    else in the prompt, and every other field of the body, is still compared to the golden byte
    for byte.
    """
    expected = json.loads(json.dumps(ts_body))  # a deep copy
    expected["stream"] = False
    for item in expected["input"]:
        item["type"] = "message"
    system = expected["input"][0]
    assert system["role"] == "system", "the golden's first input item is no longer the system turn"
    system["content"] = with_repository_scope(system["content"])
    return expected


@pytest.fixture(autouse=True)
def default_planner_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # The golden was recorded with the default model. A PLANNER_MODEL in the shell must not
    # turn this into a comparison against a different model.
    monkeypatch.delenv("PLANNER_MODEL", raising=False)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-real",
        cohere_api_key="co-test-not-real",
        database_url="postgresql://u:p@aws-0-eu-west-1.pooler.supabase.com:5432/postgres",
    )


class Recorder:
    """The network, replaced: records each request and answers with the golden's canned reply."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(200, json=GOLDEN["meta"]["cannedReply"])


async def run_python_planner(
    case: dict[str, Any], recorder: Recorder | None = None
) -> tuple[Any, Recorder]:
    recorder = recorder or Recorder()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)) as client:
        model = openai_planner_model(settings(), http_async_client=client)
        history = [HistoryTurn(t["role"], t["text"]) for t in case["history"]]
        plan = await plan_query(case["question"], history, planner=build_planner(model))
    return plan, recorder


def sha256(path: str) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


def test_the_golden_matches_the_current_typescript() -> None:
    assert sha256("lib/plan.ts") == GOLDEN["meta"]["planTsSha256"], (
        f"lib/plan.ts changed after the golden file was written; {REGENERATE}"
    )
    assert sha256("evals/planner-cases.ts") == GOLDEN["meta"]["casesTsSha256"], (
        f"evals/planner-cases.ts changed after the golden file was written; {REGENERATE}"
    )


def test_the_python_prompt_is_the_typescript_prompt_plus_one_paragraph() -> None:
    """The re-framed pin, stated directly rather than only through a request body (step 3.5b).

    Three claims, and the third is the one that stops this becoming a fork. Delete the paragraph
    from what Python sends and you are back at the TypeScript prompt exactly: no reworded rule, no
    extra sentence that crept in beside it, nothing removed.
    """
    golden_system = CASES[0]["request"]["body"]["input"][0]["content"]
    assert golden_system == TS_SYSTEM_PROMPT, (
        f"the golden's system prompt is no longer lib/plan.ts's; {REGENERATE}"
    )
    assert SYSTEM_PROMPT.count(REPOSITORY_SCOPE_PARAGRAPH) == 1
    assert SYSTEM_PROMPT.replace(f"\n\n{REPOSITORY_SCOPE_PARAGRAPH}", "", 1) == TS_SYSTEM_PROMPT


def test_every_case_carries_the_same_system_prompt() -> None:
    # The claim above is made from case 0. It generalises only because the recorder froze one
    # prompt for all 23; if that ever stops being true, the pin is testing one case out of 23.
    assert {c["request"]["body"]["input"][0]["content"] for c in CASES} == {TS_SYSTEM_PROMPT}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_same_request_as_typescript(case: dict[str, Any]) -> None:
    plan, recorder = await run_python_planner(case)

    # One call: no retry, no second request.
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert (request.method, str(request.url)) == (case["request"]["method"], case["request"]["url"])

    sent = json.loads(request.content)
    expected = encoded_by_langchain(case["request"]["body"])
    differing = sorted(k for k in sent.keys() | expected.keys() if sent.get(k) != expected.get(k))
    assert sent == expected, f"request differs from TypeScript in: {differing}"

    # The canned reply parsed (a fallback would also have made exactly one request).
    assert plan.intent == "search"
    assert plan.queries == (SubQuery(query="golden query", hypothetical="golden hypothetical"),)
    assert plan.usage == TokenUsage(input_tokens=1000, output_tokens=50)


class FailingServer(Recorder):
    """Answers every request with a 500. retry-after-ms makes the client's wait 1 ms instead of
    its 0.5 s and 1 s backoff, so the test does not sleep."""

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(
            500,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "boom", "type": "server_error", "code": "server_error"}},
        )


async def test_a_server_error_is_retried_twice_then_falls_back() -> None:
    # Three attempts, like the AI SDK's generateText (maxRetries 2), then the raw-question
    # fallback with no usage. Pins max_retries: ChatOpenAI would otherwise defer to the client.
    case = next(c for c in CASES if c["id"] == "expand-sdk")
    plan, server = await run_python_planner(case, FailingServer())
    assert len(server.requests) == 1 + PLANNER_MAX_RETRIES == 3
    assert plan.intent == "search"
    assert plan.queries == (SubQuery(query=case["question"], hypothetical=""),)
    assert plan.usage == TokenUsage(input_tokens=None, output_tokens=None)
