import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# RAG imports construct a Gemini client at import time; tests mock the only
# generation function and never make an external call.
os.environ.setdefault("GEMINI_API_KEY", "unit-test-placeholder")

import main
from services import rag_service, research_pipeline, research_tools


class FixedRAGPipelineTests(unittest.TestCase):
    def test_normal_request_uses_one_gemini_call_and_preserves_response_schema(self):
        papers = [
            {
                "openalex_id": f"https://openalex.org/W{index}",
                "title": f"Paper {index}",
                "authors": ["Researcher"],
                "year": 2025,
                "abstract": f"Abstract {index}",
                "url": f"https://doi.org/example{index}",
                "pdf_url": f"https://example.org/paper{index}.pdf" if index < 5 else None,
            }
            for index in range(1, 6)
        ]
        query_vector = np.zeros(384, dtype=np.float32)
        chunks = [f"Source: Paper {index}\nExcerpt: relevant evidence {index}" for index in range(5)]

        with (
            patch.object(
                research_tools,
                "search_papers",
                return_value={"success": True, "papers": papers},
            ),
            patch.object(research_pipeline, "create_embeddings", return_value=[query_vector]),
            patch.object(research_tools, "retrieve_papers", return_value=[{"success": True}] * 3) as pdfs,
            patch.object(
                research_tools,
                "retrieve_relevant_content",
                return_value={"chunks": chunks, "query_embedding": query_vector},
            ) as retrieval,
            patch.object(
                rag_service,
                "generate_content",
                return_value=SimpleNamespace(text="Grounded research answer [S1]."),
            ) as gemini_call,
        ):
            result = research_pipeline.run_research("How does method X work?")

        self.assertEqual(gemini_call.call_count, 1)
        self.assertEqual(gemini_call.call_args.kwargs["model"], "gemini-3-flash-preview")
        self.assertEqual(pdfs.call_args.args[0], papers[:3])
        self.assertEqual(retrieval.call_args.kwargs["top_k"], 5)
        self.assertEqual(len(result["retrieved_chunks"]), 5)
        self.assertEqual(
            set(result),
            {
                "success",
                "query",
                "answer",
                "papers",
                "sources",
                "retrieved_chunks",
                "agent_actions",
                "research_plan",
                "iterations",
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Grounded research answer [S1].")

    def test_post_endpoint_keeps_existing_response_shape(self):
        expected = {
            "success": True,
            "query": "research question",
            "answer": "answer",
            "papers": [],
            "sources": [],
            "retrieved_chunks": [],
            "agent_actions": [],
            "research_plan": [],
            "iterations": 1,
        }
        with patch.object(main, "run_research", return_value=expected):
            result = main.research_post({"query": "research question"})
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
