"""Generation's pure parts: score formatting, the system prompt, the message list, the model's
settings. No network. The request as a whole is compared with TypeScript in
test_generation_request_parity.py; these pin the pieces, so a failure there points somewhere.
"""

import pytest

from copilot_agent.generation import (
    NO_CONTEXT,
    build_system_prompt,
    finish_reason,
    generation_messages,
    js_to_fixed,
    openai_generation_model,
)
from copilot_agent.planner import HistoryTurn
from copilot_agent.refusal import REFUSAL_MESSAGE
from copilot_agent.retrieval import RetrievedChunk
from copilot_agent.settings import Settings


def chunk(content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=content, title="T", source_url="https://ai-sdk.dev/x", score=score
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The two exact halfway values in the golden, where half-to-even (Python's format)
        # and half-away-from-zero (JavaScript's toFixed) part ways:
        pytest.param(0.125, "0.13", id="0.125"),
        pytest.param(0.625, "0.63", id="0.625"),
        pytest.param(0.375, "0.38", id="0.375-agrees-anyway"),
        pytest.param(0.005, "0.01", id="0.005-stored-above-half"),
        pytest.param(0.0049999, "0.00", id="below-half"),
        pytest.param(0.3, "0.30", id="threshold"),
        pytest.param(1.0, "1.00", id="one"),
        pytest.param(-0.125, "-0.13", id="negative-away-from-zero"),
        pytest.param(-0.0, "0.00", id="negative-zero"),
    ],
)
def test_js_to_fixed_matches_javascript(value: float, expected: str) -> None:
    # Expected values checked with node: (0.125).toFixed(2) etc., 2026-09-11.
    assert js_to_fixed(value, 2) == expected


def test_python_format_is_not_enough() -> None:
    # Why js_to_fixed exists: the obvious port prints a different prompt for this score.
    assert f"{0.125:.2f}" == "0.12" != js_to_fixed(0.125, 2)


def test_no_chunks_says_so_and_quotes_the_refusal_once() -> None:
    prompt = build_system_prompt([])
    assert prompt.endswith("DOCUMENTATION:\n" + NO_CONTEXT)
    assert prompt.count(REFUSAL_MESSAGE) == 1


def test_chunks_are_numbered_from_one_and_separated() -> None:
    prompt = build_system_prompt([chunk("first", 0.759), chunk("second", 0.3)])
    assert prompt.endswith(
        "DOCUMENTATION:\n"
        "[Source 1] (relevance: 0.76)\nfirst"
        "\n\n---\n\n"
        "[Source 2] (relevance: 0.30)\nsecond"
    )


def test_braces_in_the_docs_are_text_not_placeholders() -> None:
    prompt = build_system_prompt([chunk("const x = {context} and {refusal_message} and {0}", 1)])
    assert prompt.endswith("const x = {context} and {refusal_message} and {0}")


def test_the_final_turn_is_the_resolved_sub_queries() -> None:
    history = [HistoryTurn("user", "how do I stream text"), HistoryTurn("assistant", "Use it.")]
    messages = generation_messages([], history, "and tools?", ["What is A?", "What is B?"])
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert messages[-1].content == [{"type": "text", "text": "What is A?\nWhat is B?"}]
    assert messages[1].content == [{"type": "text", "text": "how do I stream text"}]
    assert messages[2].content == [{"type": "text", "text": "Use it."}]


def test_without_sub_queries_the_raw_question_is_answered() -> None:
    messages = generation_messages([], [], "the raw question", [])
    assert messages[-1].content == [{"type": "text", "text": "the raw question"}]


def settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-real",
        cohere_api_key="co-test-not-real",
        database_url="postgresql://u:p@aws-0-eu-west-1.pooler.supabase.com:5432/postgres",
        **overrides,
    )


def test_model_settings_follow_settings() -> None:
    model = openai_generation_model(settings(generation_model="gpt-4.1-mini", max_output_tokens=77))
    assert (model.model_name, model.max_tokens) == ("gpt-4.1-mini", 77)
    assert (model.temperature, model.streaming, model.use_responses_api) == (0, True, True)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param({"status": "completed"}, "stop", id="completed"),
        pytest.param({}, "stop", id="no-metadata"),
        pytest.param({"incomplete_details": None}, "stop", id="null-details"),
        pytest.param(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "length",
            id="max-output-tokens",
        ),
        pytest.param(
            {"incomplete_details": {"reason": "content_filter"}}, "content-filter", id="filter"
        ),
        pytest.param({"incomplete_details": {"reason": "something_new"}}, "other", id="unknown"),
    ],
)
def test_finish_reason_maps_like_ai_sdk_openai(metadata: dict, expected: str) -> None:
    assert finish_reason(metadata) == expected
