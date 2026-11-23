"""Deterministic OpenAlex, PDF, and retrieval-augmented research pipeline."""

import logging
import os
import re
import time
import uuid

from services.embedding_service import create_embeddings
from services import rag_service
from services import research_tools
from services.paper_search import MAX_OPENALEX_PAPERS
from services.timing import log_process_rss, measure_stage

logger = logging.getLogger(__name__)


def _safe_error_text(value) -> str:
    """Render an exception field while redacting credentials before logging."""
    text = str(value)
    secret_markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    secrets = sorted(
        (value for name, value in os.environ.items()
         if value and any(marker in name.upper() for marker in secret_markers)),
        key=len,
        reverse=True,
    )
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", text)
    text = re.sub(
        r"(?i)(\b(?:key|api[_-]?key|access[_-]?token|client[_-]?secret|password)\b\s*[=:]\s*)[^\s,}\"']+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", text)
    return text
MAX_PAPERS_TO_PROCESS = MAX_OPENALEX_PAPERS
TOP_RETRIEVED_CHUNKS = 5


def run_research(query: str) -> dict:
    """Run search, bounded PDF indexing, retrieval, and one final RAG synthesis."""
    request_started = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    log_process_rss("request_start", request_id, elapsed_seconds="0.000")
    actions = []

    def mark(status, tool):
        actions.append({"status": status, "tool": tool})

    plan = [
        "Search OpenAlex and retrieve paper metadata and abstracts",
        "Process up to three available open PDFs using the paper cache",
        "Retrieve the five most relevant indexed research chunks",
        "Generate one answer grounded in the retrieved evidence",
    ]

    mark("Searching OpenAlex", "search_papers")
    try:
        with measure_stage("OpenAlex search and paper metadata", request_id):
            search_result = research_tools.search_papers(
                query,
                limit=MAX_OPENALEX_PAPERS,
                request_id=request_id,
            )
    except Exception as exc:
        logger.error(
            "[RAG][%s] OpenAlex search failed error_type=%s",
            request_id,
            type(exc).__name__,
        )
        search_result = {"success": False, "papers": []}
    papers = (search_result.get("papers", []) if search_result.get("success") else [])[
        :MAX_OPENALEX_PAPERS
    ]

    query_embedding = None
    try:
        query_embedding = create_embeddings([query], request_id=request_id)[0]
    except Exception as exc:
        logger.error(
            "[RAG][%s] Query embedding failed error_type=%s",
            request_id,
            type(exc).__name__,
        )

    # OpenAlex already ranks papers for this query. Select the first three with
    # accessible PDFs in that order; no model makes this selection.
    selected_papers = [paper for paper in papers if paper.get("pdf_url")][:MAX_PAPERS_TO_PROCESS]
    paper_results = []
    if selected_papers:
        mark("Downloading, extracting, chunking, and indexing selected PDFs", "retrieve_papers")
        try:
            paper_results = research_tools.retrieve_papers(
                selected_papers,
                request_id=request_id,
                query_embedding=query_embedding,
            )
        except Exception as exc:
            logger.error(
                "[RAG][%s] PDF processing failed error_type=%s",
                request_id,
                type(exc).__name__,
            )

    retrieval_result = {"chunks": [], "query_embedding": query_embedding}
    if query_embedding is not None:
        mark("Retrieving top relevant research chunks", "retrieve_relevant_content")
        try:
            retrieval_result = research_tools.retrieve_relevant_content(
                query,
                top_k=TOP_RETRIEVED_CHUNKS,
                request_id=request_id,
                query_embedding=query_embedding,
                papers=papers,
            )
        except Exception as exc:
            logger.error(
                "[RAG][%s] Semantic retrieval failed error_type=%s",
                request_id,
                type(exc).__name__,
            )
            retrieval_result = {
                "chunks": [],
                "query_embedding": query_embedding,
                "error": f"Semantic retrieval failed ({type(exc).__name__}).",
            }

    mark("Synthesizing research answer", "generate_answer")
    try:
        rag_result = rag_service.generate_answer(
            query,
            top_k=TOP_RETRIEVED_CHUNKS,
            request_id=request_id,
            query_embedding=query_embedding,
            retrieval_result=retrieval_result,
            papers=papers,
        )
        answer = rag_result.get("answer") or "The retrieved sources do not provide sufficient evidence to answer confidently."
        retrieved_chunks = rag_result.get("retrieved_chunks", retrieval_result.get("chunks", []))
        sources = rag_result.get("paper_sources", [])
    except Exception as exc:
        api_details = getattr(exc, "details", None)
        logger.error(
            "[RAG][%s] Final Gemini synthesis failed error_type=%s error_message=%s http_status=%s http_code=%s api_details=%s",
            request_id,
            type(exc).__name__,
            _safe_error_text(getattr(exc, "message", None) or exc),
            _safe_error_text(getattr(exc, "status", None)),
            _safe_error_text(getattr(exc, "code", None)),
            _safe_error_text(api_details) if api_details is not None else "unavailable",
        )
        answer = "The research evidence was collected, but answer synthesis is temporarily unavailable. Please try again."
        retrieved_chunks = retrieval_result.get("chunks", [])
        sources = [
            {
                "id": f"S{index}",
                "title": paper.get("title"),
                "authors": paper.get("authors") or [],
                "year": paper.get("year"),
                "url": paper.get("url"),
            }
            for index, paper in enumerate(papers, 1)
        ]

    logger.info(
        "[DIAG][%s] Request totals papers=%d selected_papers=%d papers_processed=%d chunks_retrieved=%d",
        request_id,
        len(papers),
        len(selected_papers),
        len(paper_results),
        len(retrieved_chunks),
    )
    logger.info(
        "[TIMING][%s] stage_complete name=Total request duration_seconds=%.3f",
        request_id,
        time.perf_counter() - request_started,
    )
    log_process_rss(
        "request_complete",
        request_id,
        elapsed_seconds=f"{time.perf_counter() - request_started:.3f}",
    )

    # Keep legacy response keys for the existing frontend/API contract. Actions
    # now describe fixed pipeline stages and never represent model decisions.
    return {
        "success": True,
        "query": query,
        "answer": answer,
        "papers": papers,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "agent_actions": actions,
        "research_plan": plan,
        "iterations": 1,
    }
