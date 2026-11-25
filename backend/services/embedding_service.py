"""Thread-safe lazy SentenceTransformer embeddings for the RAG pipeline."""

from threading import Lock

from sentence_transformers import SentenceTransformer

from services.timing import measure_stage

MODEL_ID = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_model = None
_model_lock = Lock()


def get_embedding_model():
    """Load the shared embedding model once, on first use."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_ID)
    return _model


def create_embeddings(texts, request_id="unknown"):
    with measure_stage("SentenceTransformer embedding generation", request_id):
        return get_embedding_model().encode(texts, convert_to_numpy=True)