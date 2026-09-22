# use the embedding model selected in the application settings
from config.settings import SELECTED_EMBEDDING_MODEL
# load the embedding model through the shared retriever helper
from rag.retriever import load_embedding_model
# keep one shared model instance so it is not loaded again on every request
_embedding_model = None
# return the cached embedding model, loading it only the first time
def get_embedding_model():
    global _embedding_model
    # lazy loading avoids using memory until the model is actually needed
    if _embedding_model is None:
        _embedding_model = load_embedding_model(SELECTED_EMBEDDING_MODEL)

    # later calls reuse the same loaded model
    return _embedding_model