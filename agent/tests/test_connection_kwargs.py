"""The pool's connection settings: pinned, because dropping one fails silently.

Without autocommit a search still works, just three round trips slower; without
prepare_threshold=None the transaction pooler works until a query runs for the 6th time.
Neither would break a unit test on its own, so these two assert them by name.
"""

import pytest

from copilot_agent.retrieval import connection_kwargs
from copilot_agent.settings import Settings

POOLER = "postgresql://postgres.ref:pw@aws-0-eu-central-1.pooler.supabase.com"


def settings_for(port: int) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-real",
        cohere_api_key="co-test-not-real",
        database_url=f"{POOLER}:{port}/postgres",
    )


@pytest.mark.parametrize("port", [5432, 6543])
def test_every_connection_is_autocommit(port: int) -> None:
    assert connection_kwargs(settings_for(port))["autocommit"] is True


def test_session_pooler_keeps_prepared_statements() -> None:
    assert "prepare_threshold" not in connection_kwargs(settings_for(5432))


def test_transaction_pooler_turns_prepared_statements_off() -> None:
    assert connection_kwargs(settings_for(6543))["prepare_threshold"] is None
