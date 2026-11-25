import unittest
from unittest.mock import patch

import numpy as np

from services import embedding_service


class EmbeddingServiceTests(unittest.TestCase):
    def test_sentence_transformer_model_is_loaded_once_and_used(self):
        embedding_service._model = None
        try:
            with patch.object(embedding_service, "SentenceTransformer") as model_class:
                model = model_class.return_value
                model.encode.return_value = np.zeros((2, 384), dtype=np.float32)
                result1 = embedding_service.create_embeddings(["one", "two"])
                result2 = embedding_service.create_embeddings(["three", "four"])

            model_class.assert_called_once_with("all-MiniLM-L6-v2")
            self.assertEqual(model.encode.call_count, 2)
            self.assertTrue(all(call.kwargs["convert_to_numpy"] for call in model.encode.call_args_list))
            self.assertEqual(result1.shape, (2, 384))
            self.assertEqual(result2.shape, (2, 384))
        finally:
            embedding_service._model = None


if __name__ == "__main__":
    unittest.main()
