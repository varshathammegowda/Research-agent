"""Lazy CPU ONNX embeddings for the existing MiniLM vector space."""

from threading import Lock
import time

from services.timing import log_process_rss, measure_stage

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Pin model graph and tokenizer to the same official Hub revision.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILENAME = "onnx/model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIMENSION = 384
INFERENCE_BATCH_SIZE = 16

_model = None
_model_lock = Lock()
_inference_lock = Lock()


def _mean_pool_and_normalize(hidden_states, attention_mask, np):
    """Match SentenceTransformers' attention-mask mean pooling + Normalize."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden_states.astype(np.float32) * mask).sum(axis=1)
    counts = np.maximum(mask.sum(axis=1), np.float32(1e-9))
    pooled = summed / counts
    norms = np.maximum(np.linalg.norm(pooled, ord=2, axis=1, keepdims=True), np.float32(1e-12))
    return (pooled / norms).astype(np.float32, copy=False)


class _ONNXEmbedder:
    def __init__(self, session, tokenizer, np):
        self.session = session
        self.tokenizer = tokenizer
        self.np = np
        self.input_names = {item.name for item in session.get_inputs()}

    def encode(self, texts):
        encodings = self.tokenizer.encode_batch([str(text) for text in texts])
        input_ids = self.np.asarray([item.ids for item in encodings], dtype=self.np.int64)
        attention_mask = self.np.asarray([item.attention_mask for item in encodings], dtype=self.np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = self.np.asarray(
                [item.type_ids for item in encodings], dtype=self.np.int64
            )

        outputs = self.session.run(None, {key: value for key, value in feeds.items() if key in self.input_names})
        hidden_states = next((value for value in outputs if getattr(value, "ndim", 0) == 3), None)
        if hidden_states is None:
            raise RuntimeError("ONNX model did not return token-level hidden states for mean pooling")
        embeddings = _mean_pool_and_normalize(hidden_states, attention_mask, self.np)
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings, got {embeddings.shape}"
            )
        return embeddings


def _get_model(request_id="unknown"):
    """Load the pinned official ONNX model and tokenizer on first use only."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                load_started = time.perf_counter()
                log_process_rss("before_embedding_model_load", request_id)
                with measure_stage("ONNX Runtime CPU imports and model load", request_id):
                    import numpy as np
                    import onnxruntime as ort
                    from huggingface_hub import hf_hub_download
                    from tokenizers import Tokenizer

                    model_path = hf_hub_download(
                        repo_id=MODEL_ID, filename=MODEL_FILENAME, revision=MODEL_REVISION
                    )
                    tokenizer_path = hf_hub_download(
                        repo_id=MODEL_ID, filename=TOKENIZER_FILENAME, revision=MODEL_REVISION
                    )
                    tokenizer = Tokenizer.from_file(tokenizer_path)
                    pad_id = tokenizer.token_to_id("[PAD]")
                    if pad_id is None:
                        raise RuntimeError("Official MiniLM tokenizer has no [PAD] token")
                    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
                    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]", direction="right")

                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    session = ort.InferenceSession(
                        model_path,
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    _model = _ONNXEmbedder(session, tokenizer, np)
                log_process_rss(
                    "after_embedding_model_load",
                    request_id,
                    elapsed_seconds=f"{time.perf_counter() - load_started:.3f}",
                    provider="CPUExecutionProvider",
                    model=MODEL_ID,
                )

    return _model


def create_embeddings(texts, request_id="unknown", diagnostic_fields=None):
    model = _get_model(request_id)
    diagnostic_fields = diagnostic_fields or {}
    embedding_started = time.perf_counter()
    embeddings = None
    try:
        with measure_stage("ONNX CPU embedding generation", request_id):
            with measure_stage("ONNX inference queue", request_id):
                _inference_lock.acquire()
            log_process_rss(
                "before_embedding",
                request_id,
                batch_count=(len(texts) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE,
                **diagnostic_fields,
            )
            batches = []
            try:
                for start in range(0, len(texts), INFERENCE_BATCH_SIZE):
                    batch = model.encode(texts[start : start + INFERENCE_BATCH_SIZE])
                    batches.append(batch)
                    log_process_rss(
                        "after_embedding",
                        request_id,
                        batch=f"{start // INFERENCE_BATCH_SIZE + 1}",
                        elapsed_seconds=f"{time.perf_counter() - embedding_started:.3f}",
                        **diagnostic_fields,
                    )
                embeddings = (
                    model.np.concatenate(batches, axis=0)
                    if batches
                    else model.np.empty((0, EMBEDDING_DIMENSION), dtype=model.np.float32)
                )
                return embeddings
            finally:
                batches.clear()
                _inference_lock.release()
                log_process_rss(
                    "after_embedding_cleanup",
                    request_id,
                    elapsed_seconds=f"{time.perf_counter() - embedding_started:.3f}",
                    **diagnostic_fields,
                )
    finally:
        # Keep the returned array alive for the caller; this collection only drops local references.
        if embeddings is not None:
            del embeddings
