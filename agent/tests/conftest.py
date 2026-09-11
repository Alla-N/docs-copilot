"""Fixtures shared by every test in tests/.

pytest loads this file by name. A test uses a fixture by naming it as a parameter, with no
import, which matters here: under --import-mode=importlib, test files cannot import each other
or this file.
"""

from collections.abc import Callable

import pytest

from copilot_agent.retrieval import Candidate


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Run async tests on asyncio only.

    The anyio pytest plugin runs every async test once per INSTALLED backend: asyncio today,
    trio as well the day it arrives as someone's dependency. The service is asyncio code
    (asyncio.sleep in the rerank retry, and LangGraph next), so trio is not a target.
    LangGraph's own test suite pins it the same way.

    Module scope, like the plugin's own default: an async fixture can only be as wide as this
    one, and step 1e wants a module-wide one (a single DB pool shared by the 12 parity cases).
    """
    return "asyncio"


MakeCandidate = Callable[..., Candidate]


@pytest.fixture
def make_candidate() -> MakeCandidate:
    """A factory, not a single object: tests need several candidates that differ in one field.

    Pass similarity as a LITERAL. Computed scores drift: 0.6 - 0.15 is 0.44999999999999996,
    which is below the 0.45 fallback gate, so a boundary test built that way tests nothing.
    """
    count = 0

    def make(similarity: float, *, title: str | None = None) -> Candidate:
        nonlocal count
        count += 1
        return Candidate(
            content=f"chunk {count} text",
            title=title or f"Page {count}",
            source_url=f"https://ai-sdk.dev/docs/page-{count}",
            similarity=similarity,
        )

    return make
