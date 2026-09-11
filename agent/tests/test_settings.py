"""Settings: required variables, secret hygiene, and the Supabase host guard.

No network and no real secrets. Every test builds its own fake environment, and
_env_file=None keeps the developer's real .env.local out of it.
"""

import pytest
from pydantic import ValidationError

from copilot_agent.settings import Settings

PASSWORD = "hunter2-not-real"
POOLER_URL = (
    f"postgresql://postgres.abcref:{PASSWORD}@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A complete, valid fake environment. Each test starts here and breaks one thing.

    monkeypatch undoes every setenv/delenv after the test, so tests cannot leak into each other.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("COHERE_API_KEY", "co-test-not-real")
    monkeypatch.setenv("DATABASE_URL", POOLER_URL)
    monkeypatch.delenv("VECTOR_CANDIDATES", raising=False)
    monkeypatch.delenv("RERANK_TOP_N", raising=False)
    monkeypatch.delenv("ENABLE_SEARCH_ENDPOINT", raising=False)
    return monkeypatch


def load() -> Settings:
    return Settings(_env_file=None)


def test_loads_a_complete_environment(env: pytest.MonkeyPatch) -> None:
    settings = load()
    assert settings.openai_api_key.get_secret_value() == "sk-test-not-real"
    assert settings.vector_candidates == 100
    assert settings.rerank_top_n == 5


def test_missing_variable_fails_with_its_name(env: pytest.MonkeyPatch) -> None:
    env.delenv("COHERE_API_KEY")
    with pytest.raises(ValidationError, match="cohere_api_key") as error:
        load()
    # By default pydantic echoes the input next to the error: here, the dict of every OTHER
    # variable it was given, API keys included (truncated to about 50 characters, so a leak is
    # partial and depends on value lengths, which is why this checks the marker, not a key).
    assert "input_value" not in str(error.value)


def test_secrets_stay_out_of_repr_and_str(env: pytest.MonkeyPatch) -> None:
    settings = load()
    for text in (repr(settings), str(settings)):
        assert PASSWORD not in text
        assert "sk-test-not-real" not in text


def test_direct_supabase_host_is_rejected_without_leaking_the_password(
    env: pytest.MonkeyPatch,
) -> None:
    # Short on purpose: under pydantic's ~50-character truncation the password would show.
    env.setenv("DATABASE_URL", f"postgresql://u:{PASSWORD}@db.x.supabase.co/d")
    with pytest.raises(ValidationError, match="Session pooler") as error:
        load()
    assert PASSWORD not in str(error.value)
    assert "input_value" not in str(error.value)


def test_non_postgres_url_is_rejected(env: pytest.MonkeyPatch) -> None:
    env.setenv("DATABASE_URL", "https://example.supabase.co")
    with pytest.raises(ValidationError, match="postgresql://"):
        load()


@pytest.mark.parametrize(("port", "expected"), [(5432, False), (6543, True)])
def test_detects_the_transaction_pooler(env: pytest.MonkeyPatch, port: int, expected: bool) -> None:
    env.setenv("DATABASE_URL", POOLER_URL.replace(":5432/", f":{port}/"))
    assert load().uses_transaction_pooler is expected


def test_env_overrides_retrieval_depth_like_the_typescript_side(env: pytest.MonkeyPatch) -> None:
    env.setenv("VECTOR_CANDIDATES", "40")  # env values are strings; pydantic converts
    assert load().vector_candidates == 40


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "5000"])
def test_rejects_nonsense_retrieval_depth(env: pytest.MonkeyPatch, bad: str) -> None:
    env.setenv("VECTOR_CANDIDATES", bad)
    with pytest.raises(ValidationError, match="vector_candidates"):
        load()


def test_search_endpoint_is_off_by_default(env: pytest.MonkeyPatch) -> None:
    # Off unless asked for: the paid debug route must not appear in a deployment by omission.
    assert load().enable_search_endpoint is False


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("true", True), ("0", False)])
def test_search_endpoint_flag_parses(env: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    env.setenv("ENABLE_SEARCH_ENDPOINT", raw)
    assert load().enable_search_endpoint is expected


def test_search_endpoint_flag_typo_fails_startup(env: pytest.MonkeyPatch) -> None:
    # A typo must stop the service, not silently read as off (or on).
    env.setenv("ENABLE_SEARCH_ENDPOINT", "ture")
    with pytest.raises(ValidationError, match="enable_search_endpoint"):
        load()
