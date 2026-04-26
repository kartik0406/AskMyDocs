from dotenv import load_dotenv
load_dotenv()

from elasticsearch import Elasticsearch
from pinecone import Pinecone
from google import genai
from config import *
import logging
from uuid import uuid4

logging.basicConfig(level=logging.INFO)

# 🔹 Elastic
es = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY
)

# 🔹 Pinecone
pc = Pinecone(api_key=PINCONE_API_KEY)
index = pc.Index(PINCONE_INDEX)

# 🔹 Gemini Client (for embeddings)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ------------------------
# GEMINI EMBEDDINGS (REAL SEMANTIC EMBEDDINGS)
# ------------------------

def gemini_embedding(text, dim=384):
    """Generate real semantic embeddings using Gemini embedding model.
    Uses output_dimensionality to match existing Pinecone index dimension.
    """
    try:
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                "output_dimensionality": dim
            }
        )
        return result.embeddings[0].values
    except Exception as e:
        logging.error(f"Gemini embedding failed: {e}")
        # Fallback: return zero vector (will get low scores, but won't crash)
        return [0.0] * dim


# ------------------------
# BM25 Search (FUZZY + FLEXIBLE)
# ------------------------

def bm25_search(query, doc_id, k=10):
    try:
        res = es.search(
            index="docs",
            size=k,
            query={
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text"],
                                "fuzziness": "AUTO",
                                "type": "most_fields",
                                "operator": "or",
                                "minimum_should_match": "30%"
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"doc_id": doc_id}}
                    ]
                }
            }
        )

        return [
            {"content": hit["_source"]["text"], "score": hit["_score"]}
            for hit in res["hits"]["hits"]
        ]

    except Exception as e:
        logging.error(f"Elastic failed: {e}")
        return None


# ------------------------
# Vector Search (GEMINI EMBEDDINGS)
# ------------------------

def vector_search(query, doc_id, k=10):
    try:
        emb = gemini_embedding(query)

        res = index.query(
            vector=emb,
            top_k=k,
            include_metadata=True,
            filter={"doc_id": doc_id}
        )

        return [
            {"content": m["metadata"]["text"], "score": m["score"]}
            for m in res["matches"]
        ]

    except Exception as e:
        logging.error(f"Pinecone failed: {e}")
        return []


# ------------------------
# Normalize scores to 0-1 range
# ------------------------

def normalize_scores(results):
    """Min-max normalize scores to [0, 1] range."""
    if not results:
        return results

    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        # All same score — give them all 1.0
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            r["score"] = (r["score"] - min_s) / (max_s - min_s)

    return results


# ------------------------
# Hybrid Search (NORMALIZED + ALPHA BLEND)
# ------------------------

def hybrid_search(query, doc_id, k=10, alpha=0.5):
    """
    Hybrid search combining BM25 (keyword) and vector (semantic) search.
    alpha = weight for vector search (0 = pure BM25, 1 = pure vector)
    """
    bm25_results = bm25_search(query, doc_id, k)
    vector_results = vector_search(query, doc_id, k)

    # Fallback if one fails
    if bm25_results is None and not vector_results:
        return []
    if bm25_results is None:
        candidates = [r["content"] for r in vector_results[:k]]
    elif not vector_results:
        candidates = [r["content"] for r in bm25_results[:k]]
    else:
        # Normalize both score sets to 0-1
        bm25_results = normalize_scores(bm25_results)
        vector_results = normalize_scores(vector_results)

        # Combine with alpha blending
        combined = {}

        for r in bm25_results:
            key = r["content"]
            combined[key] = (1 - alpha) * r["score"]

        for r in vector_results:
            key = r["content"]
            if key in combined:
                combined[key] += alpha * r["score"]
            else:
                combined[key] = alpha * r["score"]

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        candidates = [c for c, _ in sorted_results[:10]]

    # 🔥 Cross-encoder reranking via Gemini
    if candidates:
        candidates = rerank_with_gemini(query, candidates)

    return candidates[:8]


# ------------------------
# Cross-Encoder Reranking (via Gemini)
# ------------------------

def rerank_with_gemini(query, passages, top_k=8):
    """
    Use Gemini as a cross-encoder reranker.
    Scores each passage for relevance to the query (0-10).
    This replaces traditional PyTorch-based cross-encoders
    while staying within Render's 512MB memory limit.
    """
    try:
        # Build numbered passage list
        passage_list = ""
        for i, p in enumerate(passages):
            passage_list += f"\n[{i}] {p[:500]}\n"

        prompt = f"""You are a relevance scoring system. Score each passage for how relevant it is to the query.

Query: {query}

Passages:
{passage_list}

Return ONLY a JSON array of objects with "index" and "score" (0-10, where 10 = perfectly relevant).
Example: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 2}}]

Scores:"""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # Parse the JSON response
        import json
        text = response.text.strip()
        # Clean markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        scores = json.loads(text)
        # Sort by score descending
        scores = sorted(scores, key=lambda x: x.get("score", 0), reverse=True)

        # Return passages in reranked order
        reranked = []
        for s in scores[:top_k]:
            idx = s.get("index", 0)
            if 0 <= idx < len(passages):
                reranked.append(passages[idx])

        logging.info(f"Reranked {len(passages)} → {len(reranked)} passages")
        return reranked if reranked else passages[:top_k]

    except Exception as e:
        logging.error(f"Reranking failed, using original order: {e}")
        return passages[:top_k]


# ------------------------
# Index Documents
# ------------------------

def index_documents(chunks, doc_id):
    texts = [chunk.page_content for chunk in chunks]

    vectors = []

    for i, text in enumerate(texts):
        try:
            es.index(
                index="docs",
                document={"text": text, "doc_id": doc_id},
                refresh=True
            )
        except Exception as e:
            logging.error(f"Elastic indexing failed: {e}")

        # Use real Gemini embeddings
        emb = gemini_embedding(text)

        vectors.append({
            "id": f"{doc_id}_{i}_{uuid4()}",
            "values": emb,
            "metadata": {"text": text, "doc_id": doc_id}
        })

    if vectors:
        # Upsert in batches of 100 to avoid payload limits
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)

        logging.info(f"Indexed {len(vectors)} chunks with Gemini embeddings")