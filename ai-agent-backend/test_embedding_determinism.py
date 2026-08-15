"""
Diagnostic script: checks whether encoding the SAME question text twice
produces identical embedding vectors, or whether there's floating-point
drift between calls.

Run this directly (no Flask/FastAPI/DB needed):
    python test_embedding_determinism.py

What to look at in the output:
  - "IDENTICAL" -> the embedding step itself is clean. The instability
    you're seeing must be happening downstream (e.g. Postgres ORDER BY
    distance with no tiebreaker, when two chunks have near-identical
    distances).
  - "DIFFERENT" -> the embedding step itself has floating-point drift
    between calls. This would explain shifting distances against fixed
    chunk embeddings, causing borderline chunks to move in/out of the
    top-K.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "Is Vantable's IC3-IC4 expense pre-approval threshold higher or lower than the number of days production access reviews happen at Borealis"

print("Encoding the same question 5 times in a row...\n")

vectors = []
for i in range(5):
    vec = model.encode(question)
    vectors.append(vec)
    print(f"Run {i+1}: first 5 dims = {vec[:5]}")

print()

# Compare every run against run 1
baseline = vectors[0]
all_identical = True
for i, vec in enumerate(vectors[1:], start=2):
    identical = np.array_equal(baseline, vec)
    max_diff = np.max(np.abs(baseline - vec))
    status = "IDENTICAL" if identical else "DIFFERENT"
    if not identical:
        all_identical = False
    print(f"Run 1 vs Run {i}: {status} (max abs difference: {max_diff:.10f})")

print()
if all_identical:
    print("RESULT: All 5 runs produced byte-identical vectors.")
    print("-> The embedding step is deterministic on your machine.")
    print("-> The instability you observed is likely happening downstream,")
    print("   e.g. in the SQL ORDER BY distance with no tiebreaker column,")
    print("   or Postgres/pgvector index behavior on near-tied distances.")
else:
    print("RESULT: Vectors differed slightly between runs (floating-point drift).")
    print("-> The embedding step itself is not perfectly deterministic here.")
    print("-> This alone can shift computed distances to chunks near the")
    print("   MAX_CHUNK_DISTANCE/top-K cutoff, causing borderline chunks")
    print("   to appear or disappear from stage-1 retrieval between runs.")