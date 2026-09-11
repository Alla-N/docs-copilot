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

While the TypeScript retrieval still exists, `tests/test_ts_parity.py` keeps the calibrated
numbers identical on both sides.
