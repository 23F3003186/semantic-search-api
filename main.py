import os
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from docs import documents

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)
app = FastAPI()



class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3


def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------- Embedding Cache ----------
if os.path.exists("embed_cache.npy"):
    print("Loading cached embeddings...")
    doc_vectors = np.load("embed_cache.npy")

else:
    print("Computing document embeddings...")

    vectors = []
    for doc in documents:
        vec = get_embedding(doc["content"])
        vectors.append(vec)

    doc_vectors = np.array(vectors)
    np.save("embed_cache.npy", doc_vectors)

    print("Embeddings saved to cache.")


def rerank_with_llm(query, docs):
    scores = []

    for d in docs:
        prompt = f"""
Query: {query}
Document: {d['content']}

Rate relevance from 0-10.
Respond with only the number.
"""

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        try:
            val = float(resp.output_text.strip())
        except:
            val = 5.0

        scores.append(val / 10.0)

    return scores



def vector_search(query_vec, k):
    sims = []

    for i, vec in enumerate(doc_vectors):
        sims.append((i, cosine(query_vec, vec)))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


@app.post("/search")
def search(req: SearchRequest):

    start = time.time()

    query_vec = get_embedding(req.query)

    top = vector_search(query_vec, req.k)

    candidates = []
    for idx, score in top:
        candidates.append({
            "id": documents[idx]["id"],
            "content": documents[idx]["content"],
            "score": max(0.0, min(1.0, score)),
            "metadata": {"source": "api_docs"}
        })

    reranked = False

    # ---------- Re-Ranking ----------
    if req.rerank and len(candidates) > 0:

        reranked = True

        llm_scores = rerank_with_llm(req.query, candidates)

        for i in range(len(candidates)):
            candidates[i]["score"] = llm_scores[i]

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }

