from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank(query, docs):
    if not docs:
        return []

    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked]