from threading import Lock
import time

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
                load_started = time.perf_counter()
                log_process_rss("before_embedding_model_load", request_id)
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
                log_process_rss(
                    "after_embedding_model_load",
                    request_id,
                    elapsed_seconds=f"{time.perf_counter() - load_started:.3f}",
                )

    return _model


def create_embeddings(texts, request_id="unknown", diagnostic_fields=None):
    model = _get_model(request_id)
    diagnostic_fields = diagnostic_fields or {}
    with measure_stage("SentenceTransformer inference queue", request_id):
        _inference_lock.acquire()
    embedding_started = time.perf_counter()
    log_process_rss("before_embedding", request_id, **diagnostic_fields)
    try:
        with measure_stage("SentenceTransformer embedding generation", request_id):
            embeddings = model.encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        log_process_rss(
            "after_embedding",
            request_id,
            elapsed_seconds=f"{time.perf_counter() - embedding_started:.3f}",
            **diagnostic_fields,
        )
        return embeddings
    finally:
        _inference_lock.release()
        log_process_rss(
            "after_embedding_cleanup",
            request_id,
            elapsed_seconds=f"{time.perf_counter() - embedding_started:.3f}",
            **diagnostic_fields,
        )
