from pathlib import Path
import argparse, json, pickle, re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
 
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "vector_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "all-MiniLM-L6-v2"
 
def sentence_chunks(text):
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    return parts
 
def build_index():
    policies = json.loads((DATA / "policies.json").read_text())
    chunks = []
    for d in policies:
        for i, sentence in enumerate(sentence_chunks(d["text"])):
            chunks.append({
                "chunk_id": f'{d["doc_id"]}_C{i+1}',
                "doc_id": d["doc_id"],
                "title": d["title"],
                "text": sentence,
            })
    encoder = SentenceTransformer(MODEL_NAME)
    emb = encoder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_DIR / "policies.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f: pickle.dump(chunks, f)
    print("Indexed chunks:", len(chunks))
 
def retrieve(query, k=3):
    index = faiss.read_index(str(INDEX_DIR / "policies.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f: chunks = pickle.load(f)
    encoder = SentenceTransformer(MODEL_NAME)
    q = encoder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        out.append(item)
    return out
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()
    if args.build: build_index()
    if args.query:
        for x in retrieve(args.query, 3): print(x)