from elasticsearch import Elasticsearch
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from config import *
import logging
from uuid import uuid4
from rerank import rerank   # 🔥 IMPORTANT

logging.basicConfig(level=logging.INFO)

# 🔹 Elastic
es = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY
)

# 🔹 Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# 🔹 Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

# ------------------------
# Embeddings
# ------------------------

def get_embeddings(texts):
    return embedding_model.embed_documents(texts)

def get_query_embedding(query):
    return embedding_model.embed_query(query)

# ------------------------
# BM25 Search
# ------------------------

def bm25_search(query, doc_id, k=15):
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
        logging.error(f"❌ Elastic failed: {e}")
        return None

# ------------------------
# Vector Search
# ------------------------

def vector_search(query, doc_id, k=15):
    try:
        emb = get_query_embedding(query)

        res = index.query(
            vector=emb,
            top_k=k,
            include_metadata=True,
            filter={"doc_id": doc_id}
        )

        if not res.matches:
            return []

        return [
            {"content": m["metadata"]["text"], "score": m["score"]}
            for m in res["matches"]
        ]

    except Exception as e:
        logging.error(f"❌ Pinecone failed: {e}")
        return []

# ------------------------
# Normalize
# ------------------------

def normalize(scores):
    if not scores:
        return []

    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        return [1.0] * len(scores)

    return [(s - min_s) / (max_s - min_s) for s in scores]

# ------------------------
# Hybrid Search + Rerank
# ------------------------

def hybrid_search(query, doc_id, k=15, alpha=0.5):
    bm25_results = bm25_search(query, doc_id, k)
    vector_results = vector_search(query, doc_id, k)

    if not bm25_results and not vector_results:
        return []

    if bm25_results is None:
        logging.warning("⚠️ Using VECTOR ONLY fallback")
        return [r["content"] for r in vector_results[:k]]

    try:
        bm25_scores = normalize([r["score"] for r in bm25_results])
        vector_scores = normalize([r["score"] for r in vector_results])

        combined = {}

        for i, r in enumerate(bm25_results):
            combined[r["content"]] = alpha * bm25_scores[i]

        for i, r in enumerate(vector_results):
            if r["content"] in combined:
                combined[r["content"]] += (1 - alpha) * vector_scores[i]
            else:
                combined[r["content"]] = (1 - alpha) * vector_scores[i]

        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = [c for c, _ in sorted_results[:k]]

        # 🔥 RERANK HERE
        results = rerank(query, results)

        return results[:5]

    except Exception as e:
        logging.error(f"❌ Hybrid merge failed: {e}")
        return [r["content"] for r in vector_results[:k]]

# ------------------------
# Index Documents
# ------------------------

def index_documents(chunks, doc_id):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(texts)

    vectors = []

    for i, text in enumerate(texts):
        try:
            es.index(
                index="docs",
                document={"text": text, "doc_id": doc_id},
                refresh=True
            )
        except Exception as e:
            logging.error(f"❌ Elastic indexing failed: {e}")

        vectors.append({
            "id": f"{doc_id}_{i}_{uuid4()}",
            "values": embeddings[i],
            "metadata": {"text": text, "doc_id": doc_id}
        })

    if vectors:
        index.upsert(vectors=vectors)
        logging.info(f"✅ Indexed {len(vectors)} chunks")