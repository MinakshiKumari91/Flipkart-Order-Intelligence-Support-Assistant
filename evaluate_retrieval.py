from pathlib import Path
import json
from part3_rag import retrieve
 
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
 
answer_key = json.loads((DATA / "retrieval_answer_key.json").read_text())
rows = []
for item in answer_key:
    hits = retrieve(item["query"], k=10)
    # Deduplicate chunks to parent documents, preserving rank, until top 3 docs are obtained.
    retrieved_docs = []
    for h in hits:
        if h["doc_id"] not in retrieved_docs:
            retrieved_docs.append(h["doc_id"])
        if len(retrieved_docs) == 3:
            break
    relevant = set(item["relevant_docs"])
    retrieved_set = set(retrieved_docs)
    tp = len(relevant & retrieved_set)
    p3 = tp / 3
    r3 = tp / len(relevant)
    row = {
        "query": item["query"],
        "relevant_docs": item["relevant_docs"],
        "retrieved_docs_at_3": retrieved_docs,
        "true_positives": tp,
        "precision_at_3": p3,
        "recall_at_3": r3,
        "arithmetic": f"P@3={tp}/3; R@3={tp}/{len(relevant)}",
    }
    rows.append(row)
 
summary = {
    "per_query": rows,
    "average_precision_at_3": sum(x["precision_at_3"] for x in rows) / len(rows),
    "average_recall_at_3": sum(x["recall_at_3"] for x in rows) / len(rows),
}
(RESULTS / "retrieval_metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))