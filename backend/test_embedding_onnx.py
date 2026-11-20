import unittest

import numpy as np

from services.embedding_service import _ONNXEmbedder, _mean_pool_and_normalize


class _FakeEncoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask
        self.type_ids = [0] * len(ids)


class _FakeTokenizer:
    def encode_batch(self, texts):
        return [_FakeEncoding([1, 2, 0], [1, 1, 0]) for _ in texts]


class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def get_inputs(self):
        return [_FakeInput("input_ids"), _FakeInput("attention_mask"), _FakeInput("token_type_ids")]

    def run(self, _, feeds):
        self.feeds = feeds
        # Same token vector at each active position; padded position is masked.
        hidden = np.zeros((1, 3, 384), dtype=np.float32)
        hidden[0, :2, :2] = [3.0, 4.0]
        hidden[0, 2] = 90.0
        return [hidden]


class ONNXEmbeddingTests(unittest.TestCase):
    def test_mean_pooling_masks_padding_and_normalizes(self):
        hidden = np.array([[[3.0, 4.0], [3.0, 4.0], [90.0, 90.0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        result = _mean_pool_and_normalize(hidden, mask, np)
        np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-6)

    def test_onnx_embedder_feeds_expected_inputs_and_returns_float32(self):
        session = _FakeSession()
        embedder = _ONNXEmbedder(session, _FakeTokenizer(), np)
        result = embedder.encode(["example"])
        np.testing.assert_allclose(result[0, :2], [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(result[0, 2:], np.zeros(382), atol=1e-6)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(set(session.feeds), {"input_ids", "attention_mask", "token_type_ids"})


if __name__ == "__main__":
    unittest.main()
