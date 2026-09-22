
# used to create a stable fingerprint for the resource collection
import hashlib
# used to save and load FAISS metadata
import json
# used to manage the local FAISS storage folder and files
from pathlib import Path
# used for embedding arrays, similarity scoring and result ordering
import numpy as np
# used to search through stored information quickly based on similarity
import faiss


# local files used to store the FAISS index and its metadata
FAISS_FOLDER = Path(__file__).resolve().parent / "faiss_store"
INDEX_FILE = FAISS_FOLDER / "resources.index"
META_FILE = FAISS_FOLDER / "resources_meta.json"

# keeps the current index in memory so it can be reused
_cached_index = None
_cached_fingerprint = None

# loads the selected sentence-transformer embedding model
def load_embedding_model(model_name):
    # imported here so the model library loads only when needed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # keep the model name so the retriever knows how to format queries
    model._selected_model_name = model_name
    return model


# combines the main resource fields into one searchable text string
def build_resource_text(resource):
    title = resource.get("title", "")
    topics = resource.get("topics", [])
    content = resource.get("content", "")
    # topics may already be stored as text or as a list
    if isinstance(topics, str):
        topic_text = topics
    else:
        topic_text = " ".join(topics)
    return f"{title}\n{topic_text}\n{content}".strip()


# formats a search query based on the requirements of the embedding model
def format_query(query, model_name):
    # bge models use a retrieval instruction before the query
    if model_name and model_name.startswith("BAAI/bge"):
        return (
            "Represent this sentence for searching relevant passages: "+ query)
    # e5 models distinguish queries using the query prefix
    if model_name and model_name.startswith("intfloat/e5"):
        return "query: " + query
    # other models can use the original query directly
    return query


# formats resource text when the selected model expects a passage prefix
def format_resources(texts, model_name):
    if model_name and model_name.startswith("intfloat/e5"):
        return ["passage: " + text for text in texts]
    return texts


# creates a hash representing the current resources and embedding model
def resource_fingerprint(resources, model_name):
    # the fingerprint changes if the model or searchable resource text changes
    data = {
        "model": model_name,
        "resources": [
            {
                "id": resource.get("id"),
                "text": build_resource_text(resource),
            }
            for resource in resources
        ],
    }

    # create a consistent json representation before hashing
    encoded = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# creates a new FAISS index from the current resource embeddings
def build_faiss_index(resources, embedding_model, model_name):
    # faiss is imported only when an index needs to be built
    import faiss
    # prepare the searchable text for every resource
    texts = [
        build_resource_text(resource)
        for resource in resources
    ]

    texts = format_resources(texts, model_name)
    # normalised embeddings allow inner product to work as cosine similarity
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # create a flat inner-product index and add all resource vectors
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)
    return index


# returns a cached or saved FAISS index when it still matches the current resources
def get_faiss_index(resources, embedding_model):
    global _cached_index, _cached_fingerprint

    # read the model name stored when the embedding model was loaded
    model_name = getattr(
        embedding_model,
        "_selected_model_name",
        None,
    )

    # calculate the fingerprint expected for the current data
    fingerprint = resource_fingerprint(
        resources,
        model_name,
    )

    # reuse the in-memory index if nothing has changed
    if (
        _cached_index is not None
        and _cached_fingerprint == fingerprint
    ):
        return _cached_index

    #import faiss

    # make sure the local FAISS storage folder exists
    FAISS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # try to reuse the saved index when its fingerprint still matches
    if INDEX_FILE.exists() and META_FILE.exists():
        try:
            metadata = json.loads(
                META_FILE.read_text(
                    encoding="utf-8"
                )
            )

            if metadata.get("fingerprint") == fingerprint:
                _cached_index = faiss.read_index(
                    str(INDEX_FILE)
                )

                _cached_fingerprint = fingerprint

                return _cached_index

        except Exception:
            # a damaged or outdated saved index will simply be rebuilt
            pass

    # build a fresh index when no valid cached version is available
    index = build_faiss_index(
        resources,
        embedding_model,
        model_name,
    )

    # save the new index so later runs do not need to rebuild embeddings
    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    META_FILE.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "model": model_name,
                "resource_count": len(resources),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # also keep the index in memory for later queries in the same session
    _cached_index = index
    _cached_fingerprint = fingerprint

    return index


# retrieves the most relevant resources for a user query
def retrieve_resources(
    query,
    resources,
    embedding_model,
    top_k=3,
):
    # empty queries cannot produce meaningful retrieval results
    if not query or not query.strip():
        return []

    # there is nothing to search when no resources or results are requested
    if not resources or top_k <= 0:
        return []

    model_name = getattr(
        embedding_model,
        "_selected_model_name",
        None,
    )

    # keep simple fake models used by tests compatible with this retriever
    if model_name is None:
        texts = [
            build_resource_text(resource)
            for resource in resources
        ]

        resource_embeddings = np.asarray(
            embedding_model.encode(
                texts,
                normalize_embeddings=True,
            )
        )

        query_embedding = np.asarray(
            embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )[0]
        )

        # calculate similarity between the query and each resource
        scores = np.dot(
            resource_embeddings,
            query_embedding,
        )

        # sort from highest similarity to lowest
        indexes = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            {
                **dict(resources[index]),
                "retrieval_score": round(
                    float(scores[index]),
                    4,
                ),
            }
            for index in indexes
        ]

    # use the cached FAISS index for the real configured embedding model
    index = get_faiss_index(
        resources,
        embedding_model,
    )

    # apply any query prefix required by the selected model
    formatted_query = format_query(
        query,
        model_name,
    )

    query_embedding = embedding_model.encode(
        [formatted_query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    # never request more results than the number of available resources
    k = min(top_k,len(resources),)

    # FAISS returns similarity scores and matching resource positions
    scores, indexes = index.search(
        query_embedding,
        k,
    )

    results = []

    # attach each similarity score to its matching resource
    for score, index_number in zip(
        scores[0],
        indexes[0],
    ):
        resource = dict(
            resources[index_number]
        )

        resource["retrieval_score"] = round(
            float(score),
            4,
        )

        results.append(resource)

    return results