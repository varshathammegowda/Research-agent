from threading import Lock
from services.timing import log_process_rss, measure_stage

_model = None
_model_lock = Lock()
_inference_lock = Lock()


def _get_model(request_id="unknown"):
    """Load SentenceTransformer only when an embedding is first requested."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                with measure_stage("SentenceTransformer imports and model load", request_id):
                    from huggingface_hub import snapshot_download
                    from huggingface_hub.errors import LocalEntryNotFoundError
                    from sentence_transformers import SentenceTransformer

                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    try:
                        cached_path = snapshot_download(model_name, local_files_only=True)
                    except LocalEntryNotFoundError:
                        # Fresh deployments still download the model normally.
                        # Cached workers avoid slow, unnecessary Hub metadata checks.
                        _model = SentenceTransformer(model_name)
                    else:
                        _model = SentenceTransformer(cached_path, local_files_only=True)
                    log_process_rss("embedding_model_loaded", request_id)

    return _model


def create_embeddings(texts, request_id="unknown"):
    model = _get_model(request_id)
    with measure_stage("SentenceTransformer inference queue", request_id):
        _inference_lock.acquire()
    try:
        with measure_stage("SentenceTransformer embedding generation", request_id):
            return model.encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
    finally:
        _inference_lock.release()
