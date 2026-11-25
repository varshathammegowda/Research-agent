import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# RAG imports construct a Gemini client at import time; tests mock the only
# generation function and never make an external call.
os.environ.setdefault("GEMINI_API_KEY", "unit-test-placeholder")

import main
from services import paper_search, rag_service, research_pipeline, research_tools, vector_store


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
                "pdf_url": f"https://example.org/paper{index}.pdf",
            }
            for index in range(1, 6)
        ]
        query_vector = np.zeros(384, dtype=np.float32)
        chunks = [
            f"Source: Paper {index % 3 + 1}\nExcerpt: relevant evidence {index}"
            for index in range(5)
        ]

        with (
            patch.object(
                research_tools,
                "search_papers",
                return_value={"success": True, "papers": papers},
            ) as openalex_search,
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
        self.assertEqual(pdfs.call_args.args[0], papers[:5])
        self.assertEqual(openalex_search.call_args.kwargs["limit"], 5)
        self.assertEqual(retrieval.call_args.kwargs["top_k"], 5)
        self.assertEqual(retrieval.call_args.kwargs["papers"], papers[:5])
        self.assertEqual(len(result["retrieved_chunks"]), 5)
        self.assertEqual(len(result["papers"]), 5)
        prompt = gemini_call.call_args.kwargs["contents"]
        self.assertIn("Paper 5", prompt)
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

    def test_openalex_request_asks_for_five_and_caps_larger_requests(self):
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"results": []},
        )
        with patch.object(paper_search.requests, "get", return_value=response) as request:
            paper_search.search_openalex("research topic")
        self.assertEqual(request.call_args.kwargs["params"]["per-page"], 5)
        with patch.object(paper_search.requests, "get", return_value=response) as request:
            paper_search.search_openalex("research topic", limit=10)
        self.assertEqual(request.call_args.kwargs["params"]["per-page"], 5)

    def test_pdf_processing_helper_caps_input_to_five_papers(self):
        papers = [{"title": f"Paper {index}"} for index in range(8)]
        with patch.object(
            research_tools,
            "retrieve_pdf",
            side_effect=lambda paper, **_: {"success": True, "title": paper["title"]},
        ) as process_pdf:
            results = research_tools.retrieve_papers(papers)
        self.assertEqual(process_pdf.call_count, 5)
        self.assertEqual([result["title"] for result in results], [f"Paper {index}" for index in range(5)])

    def test_chroma_retrieval_is_limited_to_five_selected_papers_and_five_chunks(self):
        papers = [
            {"openalex_id": f"openalex-{index}", "title": f"Paper {index}"}
            for index in range(1, 8)
        ]
        query_vector = np.zeros(384, dtype=np.float32)
        with patch.object(
            research_tools,
            "search_chunks",
            return_value={"documents": [["relevant"]], "metadatas": [[{"source": "Paper 1"}]]},
        ) as search:
            result = research_tools.retrieve_relevant_content(
                "research topic",
                top_k=5,
                query_embedding=query_vector,
                papers=papers,
            )
        self.assertEqual(search.call_args.kwargs["top_k"], 5)
        self.assertEqual(
            search.call_args.kwargs["paper_ids"],
            [research_tools._paper_identifier(paper) for paper in papers[:5]],
        )
        self.assertEqual(search.call_args.kwargs["sources"], [f"Paper {index}" for index in range(1, 6)])
        self.assertEqual(result["chunks"], ["Source: Paper 1\nExcerpt: relevant"])

    def test_chroma_query_filters_to_ids_and_legacy_titles_without_changing_top_k(self):
        paper_ids = ["openalex-1", "openalex-2", "openalex-3"]
        titles = ["Paper 1", "Paper 2", "Paper 3"]
        query_vector = np.zeros(384, dtype=np.float32)
        with patch.object(
            vector_store.collection,
            "query",
            return_value={"documents": [["chunk"]]},
        ) as query:
            vector_store.search_chunks(
                query_vector,
                top_k=5,
                paper_ids=paper_ids,
                sources=titles,
            )
        self.assertEqual(query.call_args.kwargs["n_results"], 5)
        self.assertEqual(
            query.call_args.kwargs["where"],
            {
                "$or": [
                    {"paper_id": {"$in": paper_ids}},
                    {"source": {"$in": titles}},
                ]
            },
        )

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
