"""Configuration for the agent service: one typed object, validated once.

The Python counterpart of lib/env.ts (requireEnv) and of the zod parse-then-construct
pattern: every variable the service reads is declared here with its type, and a missing or
malformed one fails at startup with its name, not three calls later inside a client.

Sources, highest priority first: real environment variables, then the repo-root .env.local
(local dev), then the defaults below. In CI, Docker and AWS there is no .env.local; the
file is simply skipped and the environment provides everything.
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from copilot_agent import retrieval_config

# agent/src/copilot_agent/settings.py -> parents[3] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Supabase's transaction pooler. It cannot keep prepared statements between transactions,
# so psycopg must be told not to create them (prepare_threshold=None).
TRANSACTION_POOLER_PORT = 6543


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env.local",
        env_file_encoding="utf-8",
        # .env.local also holds the Next.js app's variables (Upstash, signing secret, ...).
        extra="ignore",
        # A validation error otherwise prints the raw input, and the input here is secrets:
        # a bad DATABASE_URL would put the database password in the terminal and the logs.
        hide_input_in_errors=True,
    )

    # Required: no default, so a missing one fails startup with its name.
    # SecretStr keeps the value out of repr(), str() and logs; read it with .get_secret_value().
    openai_api_key: SecretStr
    cohere_api_key: SecretStr
    database_url: SecretStr  # contains the database password

    # Same env overrides as lib/retrieve.ts, same defaults (tests/test_ts_parity.py).
    vector_candidates: int = Field(default=retrieval_config.VECTOR_CANDIDATES, ge=1, le=1000)
    rerank_top_n: int = Field(default=retrieval_config.RERANK_TOP_N, ge=1, le=100)

    @field_validator("database_url")
    @classmethod
    def _must_be_a_supabase_pooler_url(cls, value: SecretStr) -> SecretStr:
        url = urlsplit(value.get_secret_value())
        if url.scheme not in {"postgresql", "postgres"}:
            raise ValueError("must be a postgresql:// connection string")
        host = url.hostname or ""
        if host.startswith("db.") and host.endswith(".supabase.co"):
            raise ValueError(
                "is Supabase's direct host, which is IPv6-only and fails from Docker and AWS. "
                "Use the Session pooler string: host ends in pooler.supabase.com, port 5432."
            )
        return value

    @property
    def uses_transaction_pooler(self) -> bool:
        return urlsplit(self.database_url.get_secret_value()).port == TRANSACTION_POOLER_PORT


@lru_cache
def get_settings() -> Settings:
    """Build Settings on first use, then return the same object.

    Lazy on purpose. A module-level `settings = Settings()` would run at import time, so any
    test that merely imports a module would need every secret set, and would crash on
    collection without them. (lib/retrieve.ts has exactly that shape: its Supabase client is
    created when the module loads.)
    """
    return Settings()
