from dotenv import load_dotenv
load_dotenv()

from elasticsearch import Elasticsearch
from pinecone import Pinecone
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
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# ------------------------
# SIMPLE TEXT EMBEDDING (LIGHTWEIGHT)
# ------------------------

def simple_embedding(text, dim=384):
    vec = [float(hash(word) % 1000) for word in text.split()]

    # 🔥 pad or trim to fixed size
    if len(vec) < dim:
        vec += [0.0] * (dim - len(vec))
    else:
        vec = vec[:dim]

    return vec
# ------------------------
# BM25 Search
# ------------------------

def bm25_search(query, doc_id, k=10):
    try:
        res = es.search(
            index="docs",
            size=k,
            query={
                "bool": {
                    "must": [
                        {"match": {"text": query}},
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
# Vector Search (SAFE)
# ------------------------

def vector_search(query, doc_id, k=10):
    try:
        emb = simple_embedding(query)

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
# Hybrid Search (NO RERANK)
# ------------------------

def hybrid_search(query, doc_id, k=10, alpha=0.6):
    bm25_results = bm25_search(query, doc_id, k)
    vector_results = vector_search(query, doc_id, k)

    if bm25_results is None:
        return [r["content"] for r in vector_results[:k]]

    combined = {}

    for r in bm25_results:
        combined[r["content"]] = r["score"]

    for r in vector_results:
        combined[r["content"]] = combined.get(r["content"], 0) + r["score"]

    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return [c for c, _ in sorted_results[:5]]

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

        vectors.append({
            "id": f"{doc_id}_{i}_{uuid4()}",
            "values": simple_embedding(text),
            "metadata": {"text": text, "doc_id": doc_id}
        })

    if vectors:
        index.upsert(vectors=vectors)
        logging.info(f"Indexed {len(vectors)} chunks")