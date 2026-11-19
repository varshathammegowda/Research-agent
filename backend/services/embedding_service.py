from threading import Lock

_model = None
_model_lock = Lock()


def _get_model():
    """Load SentenceTransformer only when an embedding is first requested."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def create_embeddings(texts):
    return _get_model().encode(
        texts,
        convert_to_numpy=True
    )
