# agent/ - docs-copilot Python agent service

The Python half of docs-copilot v2. Phase 1 ports retrieval (`search_docs`) from
`lib/retrieve.ts`; later phases add the LangGraph orchestrator behind `POST /chat`.

Managed with [uv](https://docs.astral.sh/uv/). Python 3.13 (`.python-version`).

```
cd agent
uv sync                   # create .venv from uv.lock
uv run pytest             # tests
uv run ruff check .       # lint
uv run ruff format .      # format
```

## Running the service (local)

```
uv run uvicorn --factory copilot_agent.api:create_app --reload
curl http://127.0.0.1:8000/health         # {"status":"ok"}; liveness, calls nothing

# POST /search exists only with ENABLE_SEARCH_ENDPOINT set: it spends embed + rerank credits.
ENABLE_SEARCH_ENDPOINT=1 uv run uvicorn --factory copilot_agent.api:create_app --reload
curl -s http://127.0.0.1:8000/search -H 'content-type: application/json' \
  -d '{"query": "how do I stream text"}'
```

Add `"embed_text": "..."` to embed a HyDE hypothetical instead of the query. Interactive docs
at http://127.0.0.1:8000/docs. Startup opens the database pool and exits if it cannot connect.

While the TypeScript retrieval still exists, `tests/test_ts_parity.py` keeps the calibrated
numbers identical on both sides.

## Planner (phase 2, step 2.1)

`copilot_agent/planner.py` ports `planQuery()` from `lib/plan.ts` through LangChain
(`ChatOpenAI` on the Responses API, structured output with zod's exact JSON schema). Its
parity is checked twice:

```
# exact: the Python planner sends the TypeScript planner's request (free, runs in CI)
npm run exp:planner-requests               # from the repo root; rewrites tests/golden/planner-requests.json
uv run pytest tests/test_planner_request_parity.py

# statistical: the planner eval on the real model, 23 cases x 5 runs (a few cents)
uv run python evals/planner_eval.py        # compare two runs with two of: npm run eval:planner
```

The golden stores a sha256 of `lib/plan.ts` and `evals/planner-cases.ts`, so editing either
fails the parity test until the golden is regenerated.

## Running in Docker (local)

From the repo root (the build context is `agent/`, so `.env.local` is never sent to Docker):

```
docker build -t copilot-agent agent

docker run --rm --name copilot-agent -p 127.0.0.1:8000:8000 \
  --env-file <(grep -E '^(OPENAI_API_KEY|COHERE_API_KEY|DATABASE_URL)=' .env.local) \
  copilot-agent
curl http://127.0.0.1:8000/health          # {"status":"ok"}
docker stop copilot-agent                  # SIGTERM: the lifespan closes the pool
```

Add `-e ENABLE_SEARCH_ENDPOINT=1` before the image name to get `POST /search`.

- **No secrets in the image.** They arrive as environment variables at `docker run`, and only
  the three the service reads. Not `--env-file .env.local`: that would also hand the
  container the Upstash token and the signing secret, and Docker's env-file is not dotenv, so
  a quoted value keeps its quotes.
- **`-p 127.0.0.1:8000:8000`, not `-p 8000:8000`.** The short form publishes on every
  interface of the Mac, and `/search` spends credits. Inside the container uvicorn binds
  `0.0.0.0` on purpose: Docker's port forward does not reach the container's own loopback.
- **Runs as uid 999, not root**, and cannot modify its own virtualenv.
- **A dead database fails startup** after psycopg's 10 s pool timeout, and the container exits.
- The uv version (0.12.5) is pinned in three places: the Mac, `Dockerfile` and
  `.github/workflows/agent.yml`. Bump them together.

CI (`.github/workflows/agent.yml`) runs ruff, pytest (no integration tests, no secrets) and a
`docker build` with an import smoke test on every push that touches `agent/`.
