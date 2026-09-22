# numpy is used to create fixed embedding and faiss test arrays
import numpy as np
# imports the retrieval functions being tested
from rag import retriever

# provides a small fixed resource set for retrieval tests
def resources():
    return [
        {"id": "r1", "title": "Sleep", "topics": ["sleep", "rest"], "content": "Sleep information."},
        {"id": "r2", "title": "Support", "topics": ["support"], "content": "Support information."},
        {"id": "r3", "title": "Feeding", "topics": ["feeding"], "content": "Feeding information."},
    ]


# checks that different embedding models receive the query format they expect
def test_query_format():
    assert retriever.format_query("sleep help", "BAAI/bge-small-en-v1.5").startswith("Represent this sentence")
    assert retriever.format_query("sleep help", "intfloat/e5-small-v2") == "query: sleep help"
    assert retriever.format_query("sleep help", "sentence-transformers/all-MiniLM-L6-v2") == "sleep help"
    # e5 resource text should use the passage prefix
    assert retriever.format_resources(["one"], "intfloat/e5-small-v2") == ["passage: one"]


# checks that the resource fingerprint changes only when the data or model changes
def test_fingerprint():
    items = resources()
    first = retriever.resource_fingerprint(items, "model-a")

    # an identical copy should produce the same fingerprint
    assert retriever.resource_fingerprint([dict(x) for x in items], "model-a") == first
    changed = [dict(x) for x in items]
    changed[0]["content"] = "Updated sleep information."
    # changing resource content or model name should change the fingerprint
    assert retriever.resource_fingerprint(changed, "model-a") != first
    assert retriever.resource_fingerprint(items, "model-b") != first


# checks faiss result ordering, result limits and returned similarity scores
def test_faiss_search(monkeypatch):
    # fake embedding model used so the real model does not need to load
    class Model:
        _selected_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True):
            self.calls.append(list(texts))
            return np.array([[0.25, 0.75]], dtype=np.float32)

    # fake faiss index with fixed scores and resource positions
    class Index:
        def __init__(self):
            self.k = None

        def search(self, query, k):
            self.k = k
            return (
                np.array([[0.91, 0.75, 0.50]], dtype=np.float32),
                np.array([[2, 0, 1]], dtype=np.int64),
            )

    model, index = Model(), Index()

    # replace the real faiss index with the fixed test index
    monkeypatch.setattr(retriever, "get_faiss_index", lambda items, model: index)
    result = retriever.retrieve_resources("sleep support", resources(), model, top_k=10)
    # only the three available resources should be requested
    assert index.k == 3
    # the query should be encoded once
    assert model.calls == [["sleep support"]]
    # results should follow the order returned by faiss
    assert [x["id"] for x in result] == ["r3", "r1", "r2"]
    # faiss similarity scores should be attached to each result
    assert [x["retrieval_score"] for x in result] == [0.91, 0.75, 0.50]