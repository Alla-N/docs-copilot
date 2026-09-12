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
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from copilot_agent import retrieval_config

# agent/src/copilot_agent/settings.py -> parents[3] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Supabase's transaction pooler. It cannot keep prepared statements between transactions,
# so psycopg must be told not to create them (prepare_threshold=None).
TRANSACTION_POOLER_PORT = 6543

MIN_AGENT_API_KEY_LENGTH = 32


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

    # lib/plan.ts reads PLANNER_MODEL with the same default. The planner parity test pins the
    # default: the golden request was recorded with it.
    planner_model: str = Field(default="gpt-4o-mini", min_length=1)
    # lib/generation.ts: GENERATION_MODEL and MAX_OUTPUT_TOKENS, same defaults. The cap bounds a
    # runaway answer's cost; a normal grounded answer is 300 to 600 tokens.
    generation_model: str = Field(default="gpt-4o-mini", min_length=1)
    max_output_tokens: int = Field(default=1024, ge=1, le=16384)

    # The HTTP service's two secrets. Optional HERE so the CLIs (search_cli, chat_cli) run
    # without them; api.create_app refuses to start when either is missing, so a server cannot.
    #
    # AGENT_API_KEY: the shared key the Next.js route sends as "Authorization: Bearer <key>".
    # POST /chat spends credits on every call (invariant 2), so it never answers without it.
    agent_api_key: SecretStr | None = None
    # ASSISTANT_SIGNING_SECRET: the SAME secret the Next.js route verifies history with
    # (lib/assistant-signature.ts). Python signs the answers it generates (signing.py).
    assistant_signing_secret: SecretStr | None = None

    # When LangGraph saves a /chat run's checkpoints (step 2.5): "sync" after every step, before
    # the next starts; "async" after every step, while the next runs (LangGraph's default);
    # "exit" once, when the run ends (a failed or cancelled one included). The thread's turns come
    # out the same in all three (tests/test_graph.py runs the failure and cancel cases on each).
    # "exit", because a run is never resumed halfway (a new question discards an unfinished one),
    # so the per-step saves buy nothing. Measured on Supabase (experiments/checkpoint_overhead.py,
    # 2026-09-11, 24 turns per mode, 2 repeats): per turn, exit wrote 7 rows and 9.3 KiB stored,
    # sync and async 33 rows and 25.3 KiB; before the first token exit added about 90 ms (reading
    # the thread), async 90 to 130 ms, sync about 1.1 s; to the end of the run exit added about
    # 230 ms, async about 1.2 s (its saves queue behind one lock in the saver).
    checkpoint_durability: Literal["sync", "async", "exit"] = "exit"

    # POST /search spends embed + rerank credits on every call, and invariant 2 forbids a paid
    # public endpoint. So the route is only REGISTERED when this is true (api.create_app): a
    # deployment that never sets it has no /search at all (404), rather than a /search behind
    # a runtime check someone can get wrong. Local debugging only.
    enable_search_endpoint: bool = False

    # Langfuse (step 2.7). Tracing is on only when BOTH keys are set: no keys, no traces, and the
    # service answers exactly as it did in 2.6 (tracing.py). The public key is not a secret, it
    # names the project and travels with every export; the secret key signs the export.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # Langfuse's own environment dimension: which deployment a trace came from, so the AWS
    # service (phase 2b), CI and this laptop do not share one dashboard. Langfuse's rule, not
    # ours: lowercase alphanumerics with hyphens and underscores, not starting with "langfuse".
    # (The prefix check is a validator, not part of the pattern: pydantic compiles `pattern` with
    # Rust's regex crate, which has no lookahead.)
    langfuse_environment: str = Field(default="development", pattern=r"^[a-z0-9][a-z0-9_-]*$")

    @field_validator("langfuse_base_url")
    @classmethod
    def _must_be_an_http_url(cls, value: str) -> str:
        if urlsplit(value).scheme not in {"http", "https"}:
            raise ValueError("must be an http:// or https:// URL, e.g. https://cloud.langfuse.com")
        return value

    @field_validator("langfuse_environment")
    @classmethod
    def _not_a_reserved_environment(cls, value: str) -> str:
        if value.startswith("langfuse"):
            raise ValueError('must not start with "langfuse": Langfuse reserves that prefix')
        return value

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

    @field_validator("agent_api_key")
    @classmethod
    def _key_long_enough_to_be_random(cls, value: SecretStr | None) -> SecretStr | None:
        # Not a strength test, a typo test: "changeme" or a half-pasted value fails here, at
        # startup, instead of guarding a paid endpoint. openssl rand -hex 32 gives 64 characters.
        if value is not None and len(value.get_secret_value()) < MIN_AGENT_API_KEY_LENGTH:
            raise ValueError(
                f"must be at least {MIN_AGENT_API_KEY_LENGTH} characters "
                "(generate one with: openssl rand -hex 32)"
            )
        return value

    @property
    def uses_transaction_pooler(self) -> bool:
        return urlsplit(self.database_url.get_secret_value()).port == TRANSACTION_POOLER_PORT

    def secret_values(self) -> list[str]:
        """Every secret this process holds, in the clear, for the tracing mask (tracing.py).

        The mask does not guess what a secret looks like; it is handed the values, so a span
        attribute that quotes one is exported without it. The only caller is open_tracing, and
        the values stay inside the mask's closure: nothing here is logged or returned anywhere
        else, which is the whole reason the fields are SecretStr in the first place.
        """
        held = (
            self.openai_api_key,
            self.cohere_api_key,
            self.database_url,
            self.agent_api_key,
            self.assistant_signing_secret,
            self.langfuse_secret_key,
        )
        return [secret.get_secret_value() for secret in held if secret is not None]


@lru_cache
def get_settings() -> Settings:
    """Build Settings on first use, then return the same object.

    Lazy on purpose. A module-level `settings = Settings()` would run at import time, so any
    test that merely imports a module would need every secret set, and would crash on
    collection without them. (lib/retrieve.ts has exactly that shape: its Supabase client is
    created when the module loads.)
    """
    return Settings()
