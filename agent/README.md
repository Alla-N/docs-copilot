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
