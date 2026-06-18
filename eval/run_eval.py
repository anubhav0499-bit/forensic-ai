"""
RAG Evaluation Suite — Retrieval Quality Metrics
=================================================
Tests: Recall@K, Precision@K, MRR, NDCG, Hit Rate for BM25, Dense, and Hybrid
retrievers.

References
----------
Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). "RAGAS: Automated
    Evaluation of Retrieval Augmented Generation." arXiv:2309.15217.
    Offline evaluation runner for context_relevancy, faithfulness,
    answer_relevancy, context_recall metrics.
Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long
    Contexts." Transactions of the ACL (TACL) 2024.

Role
----
Offline benchmarking script. Run as: python -m eval.run_eval --company Infosys
NOT called from the production agent pipeline. Evaluates retrieval quality against
CORPUS ground truth. Saves results to eval/results/{timestamp}.json.
"""

import sys, os, time, math, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
from eval.corpus import CORPUS, QUERY_GROUND_TRUTH
from rag.embeddings import EmbeddingModel
from rag.bm25_retriever import BM25Retriever
from rag.vector_store import VectorStore
from rag.hybrid_retriever import HybridRetriever
from processing.chunker import DocumentChunk


# ── Build chunks ──────────────────────────────────────────────────
chunks = []
for cid, ctype, section, fy, content in CORPUS:
    chunks.append(DocumentChunk(
        chunk_id=cid, content=content, chunk_type=ctype,
        source_document=f"TestCorp_AR_{fy}.pdf", section=section,
        fiscal_year=fy, word_count=len(content.split()),
    ))

content_id_map = {}
for cid, _, _, _, content in CORPUS:
    content_id_map[content[:80]] = cid   # key by first 80 chars


def result_to_ids(results: list[dict]) -> list[str]:
    """Map retrieval results back to corpus IDs via content prefix."""
    ids = []
    for r in results:
        key = r.get("content", "")[:80]
        if key in content_id_map:
            ids.append(content_id_map[key])
    return ids


# ── Metrics ───────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / min(k, len(retrieved))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    for rank, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, r in enumerate(retrieved[:k], 1)
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(retrieved: list[str], relevant: list[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


# ── Setup retrievers ──────────────────────────────────────────────
print("=" * 68)
print("RAG EVALUATION SUITE — FORENSIC AI PLATFORM")
print("=" * 68)
print()

print("[1/4] Loading embedding model...")
t0 = time.time()
embed = EmbeddingModel()
embed_load_ms = (time.time() - t0) * 1000
print(f"      {embed.model_name}  (dim={embed.get_dimension()}, load={embed_load_ms:.0f}ms)")

print("[2/4] Indexing BM25...")
t0 = time.time()
bm25 = BM25Retriever()
bm25.index_chunks("TestCorp", chunks)
bm25_index_ms = (time.time() - t0) * 1000
print(f"      {len(chunks)} chunks indexed in {bm25_index_ms:.0f}ms")

print("[3/4] Indexing ChromaDB (dense)...")
tmpdir = pathlib.Path(tempfile.mkdtemp())
t0 = time.time()
vs = VectorStore(embed, persist_dir=tmpdir)
n_added = vs.add_chunks("TestCorp", chunks)
dense_index_ms = (time.time() - t0) * 1000
print(f"      {n_added} chunks indexed in {dense_index_ms:.0f}ms")

hybrid = HybridRetriever(vs, bm25)

print("[4/4] Running 20 test queries against BM25 | Dense | Hybrid...")
print()


# ── Run evaluation ────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 10]
results_by_method = {
    "bm25":   {"queries": [], "latencies": []},
    "dense":  {"queries": [], "latencies": []},
    "hybrid": {"queries": [], "latencies": []},
}

failure_cases = []

for q_idx, (query, relevant, qtype) in enumerate(QUERY_GROUND_TRUTH):
    row = {"query": query, "relevant": relevant, "type": qtype}

    # BM25
    t0 = time.time()
    bm25_raw = bm25.search("TestCorp", query, n_results=10)
    bm25_ms = (time.time() - t0) * 1000
    bm25_ids = result_to_ids(bm25_raw)
    row_bm25 = {
        "retrieved": bm25_ids,
        "latency_ms": bm25_ms,
        **{f"p@{k}": precision_at_k(bm25_ids, relevant, k) for k in K_VALUES},
        **{f"r@{k}": recall_at_k(bm25_ids, relevant, k) for k in K_VALUES},
        "mrr": mrr(bm25_ids, relevant),
        "ndcg@5": ndcg_at_k(bm25_ids, relevant, 5),
        "hit@5": hit_rate(bm25_ids, relevant, 5),
    }

    # Dense
    t0 = time.time()
    dense_raw = vs.search("TestCorp", query, n_results=10)
    dense_ms = (time.time() - t0) * 1000
    dense_ids = result_to_ids(dense_raw)
    row_dense = {
        "retrieved": dense_ids,
        "latency_ms": dense_ms,
        **{f"p@{k}": precision_at_k(dense_ids, relevant, k) for k in K_VALUES},
        **{f"r@{k}": recall_at_k(dense_ids, relevant, k) for k in K_VALUES},
        "mrr": mrr(dense_ids, relevant),
        "ndcg@5": ndcg_at_k(dense_ids, relevant, 5),
        "hit@5": hit_rate(dense_ids, relevant, 5),
    }

    # Hybrid
    t0 = time.time()
    hybrid_raw = hybrid.search("TestCorp", query, n_results=10)
    hybrid_ms = (time.time() - t0) * 1000
    hybrid_ids = result_to_ids(hybrid_raw)
    row_hybrid = {
        "retrieved": hybrid_ids,
        "latency_ms": hybrid_ms,
        **{f"p@{k}": precision_at_k(hybrid_ids, relevant, k) for k in K_VALUES},
        **{f"r@{k}": recall_at_k(hybrid_ids, relevant, k) for k in K_VALUES},
        "mrr": mrr(hybrid_ids, relevant),
        "ndcg@5": ndcg_at_k(hybrid_ids, relevant, 5),
        "hit@5": hit_rate(hybrid_ids, relevant, 5),
    }

    results_by_method["bm25"]["queries"].append(row_bm25)
    results_by_method["bm25"]["latencies"].append(bm25_ms)
    results_by_method["dense"]["queries"].append(row_dense)
    results_by_method["dense"]["latencies"].append(dense_ms)
    results_by_method["hybrid"]["queries"].append(row_hybrid)
    results_by_method["hybrid"]["latencies"].append(hybrid_ms)

    # Collect failure cases for hybrid
    if row_hybrid["hit@5"] == 0:
        failure_cases.append({
            "q_idx": q_idx + 1,
            "query": query,
            "expected": relevant,
            "retrieved_hybrid": hybrid_ids[:5],
            "type": qtype,
        })


# ── Aggregate metrics ─────────────────────────────────────────────

def aggregate(method_data):
    qs = method_data["queries"]
    lats = method_data["latencies"]
    n = len(qs)
    return {
        "P@1":  sum(q["p@1"]  for q in qs) / n,
        "P@3":  sum(q["p@3"]  for q in qs) / n,
        "P@5":  sum(q["p@5"]  for q in qs) / n,
        "P@10": sum(q["p@10"] for q in qs) / n,
        "R@1":  sum(q["r@1"]  for q in qs) / n,
        "R@3":  sum(q["r@3"]  for q in qs) / n,
        "R@5":  sum(q["r@5"]  for q in qs) / n,
        "R@10": sum(q["r@10"] for q in qs) / n,
        "MRR":  sum(q["mrr"]  for q in qs) / n,
        "NDCG@5": sum(q["ndcg@5"] for q in qs) / n,
        "Hit@5":  sum(q["hit@5"]  for q in qs) / n,
        "lat_p50_ms": float(np.percentile(lats, 50)),
        "lat_p95_ms": float(np.percentile(lats, 95)),
        "lat_p99_ms": float(np.percentile(lats, 99)),
    }

agg = {m: aggregate(results_by_method[m]) for m in ["bm25", "dense", "hybrid"]}


# ── Print results ─────────────────────────────────────────────────

print("RETRIEVAL METRICS COMPARISON")
print("-" * 68)
print(f"{'Metric':<18} {'BM25':>10} {'Dense':>10} {'Hybrid':>10}  {'Best'}")
print("-" * 68)

metrics_to_show = [
    ("P@1",    "Precision@1"),
    ("P@3",    "Precision@3"),
    ("P@5",    "Precision@5"),
    ("P@10",   "Precision@10"),
    ("R@3",    "Recall@3"),
    ("R@5",    "Recall@5"),
    ("R@10",   "Recall@10"),
    ("MRR",    "MRR"),
    ("NDCG@5", "NDCG@5"),
    ("Hit@5",  "Hit@5"),
]

for key, label in metrics_to_show:
    b = agg["bm25"][key]
    d = agg["dense"][key]
    h = agg["hybrid"][key]
    best = "BM25" if b >= d and b >= h else ("Dense" if d >= h else "Hybrid")
    print(f"  {label:<16} {b:>10.3f} {d:>10.3f} {h:>10.3f}  {best}")

print("-" * 68)
print()
print("LATENCY (ms)")
print("-" * 68)
print(f"  {'Percentile':<16} {'BM25':>10} {'Dense':>10} {'Hybrid':>10}")
print("-" * 68)
for pct in ["lat_p50_ms", "lat_p95_ms", "lat_p99_ms"]:
    label = pct.replace("lat_", "").replace("_ms", "")
    print(f"  {label:<16} {agg['bm25'][pct]:>10.1f} {agg['dense'][pct]:>10.1f} {agg['hybrid'][pct]:>10.1f}")
print("-" * 68)
print()

# By query type
print("HYBRID PERFORMANCE BY QUERY TYPE")
print("-" * 68)
type_buckets = {}
for i, (_, _, qtype) in enumerate(QUERY_GROUND_TRUTH):
    hq = results_by_method["hybrid"]["queries"][i]
    if qtype not in type_buckets:
        type_buckets[qtype] = []
    type_buckets[qtype].append(hq)

for qtype, qrows in type_buckets.items():
    n = len(qrows)
    avg_hit  = sum(r["hit@5"] for r in qrows) / n
    avg_mrr  = sum(r["mrr"]   for r in qrows) / n
    avg_r5   = sum(r["r@5"]   for r in qrows) / n
    print(f"  {qtype:<15}  n={n}  Hit@5={avg_hit:.2f}  MRR={avg_mrr:.3f}  R@5={avg_r5:.3f}")
print("-" * 68)
print()

print("FAILURE ANALYSIS — Queries with zero Hit@5 (Hybrid)")
print("-" * 68)
if failure_cases:
    for fc in failure_cases:
        print(f"  Q{fc['q_idx']} [{fc['type']}]: {fc['query'][:70]}")
        print(f"      Expected: {fc['expected']}")
        print(f"      Got:      {fc['retrieved_hybrid']}")
        print()
else:
    print("  No zero-hit failures. All queries returned at least 1 relevant chunk.")
print("-" * 68)
print()

# Duplicate / irrelevant chunk detection
print("RETRIEVAL QUALITY — Duplicate & Irrelevant Injection Check")
print("-" * 68)
dup_count = 0
total_results = 0
for qrows in results_by_method["hybrid"]["queries"]:
    ids = qrows["retrieved"]
    total_results += len(ids)
    dups = len(ids) - len(set(ids))
    dup_count += dups
dup_rate = dup_count / total_results if total_results else 0
print(f"  Total results inspected: {total_results}")
print(f"  Duplicate chunks:        {dup_count} ({dup_rate*100:.1f}%)")

# irrelevant = retrieved but NOT in any ground-truth set
all_relevant_ids = set()
for _, relevant, _ in QUERY_GROUND_TRUTH:
    all_relevant_ids.update(relevant)
irrel = 0
for qrows in results_by_method["hybrid"]["queries"]:
    for r in qrows["retrieved"]:
        if r not in all_relevant_ids:
            irrel += 1
irrel_rate = irrel / total_results if total_results else 0
print(f"  Irrelevant injections:   {irrel} ({irrel_rate*100:.1f}%)")
print("-" * 68)
print()
print("Eval complete.")
