"""Calibrated retrieval numbers, ported from lib/retrieve.ts (and UNION_CAP from lib/plan.ts).

These are measured, not guessed (CLAUDE.md invariant 6). Move one only with two full
eval runs behind it, and move it on both sides while the TypeScript retrieval still
exists: tests/test_ts_parity.py fails when the two disagree.
"""

# Rerank score gate. Calibrated on Cohere rerank-v3.5 scores, not cosine.
RERANK_THRESHOLD = 0.30

# Gate used only when the reranker is unavailable, on raw cosine similarity (Day 6).
COSINE_THRESHOLD = 0.45

# Candidates vector search hands to the reranker. 20 -> 40 (Day 10) -> 100 (Day 15).
VECTOR_CANDIDATES = 100

# Chunks kept after reranking.
RERANK_TOP_N = 5

# Chunks the model sees after the sub-queries' results are unioned (plannedRetrieve).
UNION_CAP = 8
