"""Simple, bounded tool-using academic research agent."""
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

from .prompts import DECISION_PROMPT, SYNTHESIS_PROMPT
from .schemas import AgentDecision, ResearchState
from . import tools
from services.gemini_generation import generate_content
from services.timing import log_process_rss, measure_stage

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 5


def _status(state: ResearchState, message: str, tool: str = "") -> None:
    state.actions.append({"status": message, "tool": tool})


def _decision(client, state: ResearchState, request_id: str = "unknown") -> AgentDecision:
    papers_summary = [
        {
            "paper_index": index,
            "title": paper.get("title") or "Untitled paper",
            "year": paper.get("year"),
            "abstract": (paper.get("abstract") or "")[:500],
            "open_pdf_available": bool(paper.get("pdf_url")),
        }
        for index, paper in enumerate(state.papers[:5])
    ]
    evidence_summary = [chunk[:450] for chunk in state.chunks[:3]]
    prompt = DECISION_PROMPT.format(
        query=state.query, objective=state.objective or state.query,
        plan=json.dumps(state.plan), used=json.dumps(state.used_tools),
        paper_count=len(state.papers), chunk_count=len(state.chunks),
        papers_summary=json.dumps(papers_summary),
        evidence_summary=json.dumps(evidence_summary),
        observations=json.dumps(state.observations[-4:]),
    )
    logger.info("[DIAG][%s] Gemini decision request model=%s", request_id, "gemini-3.8-flash")
    with measure_stage("Agent decision Gemini call", request_id):
        response = generate_content(
            client,
            request_id=request_id,
            model="gemini-3.8-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
    logger.info("[DIAG][%s] Gemini decision response status=success", request_id)
    return AgentDecision.model_validate_json(response.text or "")


def _safe_error_message(error: Exception) -> str:
    message = str(error)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key|authorization)(\s*[:=]\s*)[^,\s]+",
        r"\1\2[REDACTED]",
        message,
    )
    return message[:300]


def run(client, query: str) -> dict:
    request_started = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    log_process_rss("request_started", request_id)
    diagnostics = {
        "papers_returned": 0,
        "pdfs_downloaded": 0,
        "pdfs_parsed": 0,
        "chunks_created": 0,
        "embeddings_generated": 0,
        "chunks_stored": 0,
        "chunks_retrieved": 0,
        "paper_cache_hits": 0,
    }
    with measure_stage("Query processing", request_id):
        state = ResearchState(query=query)
        state.objective = query.strip()
        state.plan = ["Find relevant papers", "Retrieve available research evidence", "Synthesize a cited answer"]
        _status(state, "Planning research")
    logger.info("[AGENT][%s] Research started", request_id)

    # Searching OpenAlex and querying the persistent local index are independent.
    # Run them together and retain the query vector for any selected PDF ingestion.
    _status(state, "Searching OpenAlex", "search_papers")
    _status(state, "Retrieving semantic research evidence", "retrieve_relevant_content")
    query_embedding = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            search_future = executor.submit(tools.search_papers, query, 5, request_id)
            retrieval_future = executor.submit(tools.retrieve_relevant_content, query, 5, request_id)
            with measure_stage("Parallel OpenAlex and local retrieval", request_id):
                try:
                    search_result = search_future.result()
                except Exception as exc:
                    logger.exception("[AGENT][%s] OpenAlex search failed", request_id)
                    search_result = {"success": False, "papers": []}
                    state.observations.append(f"OpenAlex search failed: {type(exc).__name__}")
                try:
                    retrieval_result = retrieval_future.result()
                except Exception as exc:
                    logger.exception("[AGENT][%s] Local semantic retrieval failed", request_id)
                    retrieval_result = {"chunks": [], "query_embeddings_generated": 0}
                    state.observations.append(f"Semantic retrieval failed: {type(exc).__name__}")
        if search_result.get("success"):
            state.papers = search_result.get("papers", [])
            diagnostics["papers_returned"] = len(state.papers)
            state.observations.append(f"OpenAlex returned {len(state.papers)} papers")
        else:
            state.observations.append("OpenAlex search unavailable")
        state.chunks = retrieval_result.get("chunks", [])
        diagnostics["embeddings_generated"] += retrieval_result.get("query_embeddings_generated", 0)
        diagnostics["chunks_retrieved"] = len(state.chunks)
        if retrieval_result.get("error"):
            state.observations.append(retrieval_result["error"])
        else:
            state.observations.append(f"Retrieved {len(state.chunks)} existing evidence chunks")
        query_embedding = retrieval_result.get("query_embedding")
    except Exception as exc:
        logger.exception("[AGENT][%s] Initial search or local retrieval failed", request_id)
        state.observations.append(f"Initial search or local retrieval failed: {type(exc).__name__}")
    state.used_tools.extend(["search_papers", "retrieve_relevant_content"])

    for turn in range(MAX_ITERATIONS):
        state.iteration = turn + 1
        try:
            decision = _decision(client, state, request_id)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning(
                "[DIAG][%s] Agent decision invalid model=%s error_type=%s",
                request_id,
                "gemini-3.8-flash",
                type(exc).__name__,
            )
            decision = AgentDecision(action="finish", reason="Decision could not be validated")
        except Exception as exc:
            logger.error(
                "[DIAG][%s] Agent decision failed model=%s code=%s error_type=%s error=%s",
                request_id,
                "gemini-3.8-flash",
                getattr(exc, "code", None) or getattr(exc, "status_code", None),
                type(exc).__name__,
                _safe_error_message(exc),
            )
            state.observations.append(f"Decision model unavailable: {type(exc).__name__}")
            decision = AgentDecision(action="finish")

        logger.info("[DIAG][%s] Agent selected action=%s", request_id, decision.action)
        if decision.action == "finish":
            break
        if decision.action == "search_papers" and decision.action in state.used_tools:
            state.observations.append("Paper search already completed")
            break
        if decision.action in state.used_tools and decision.action in ("retrieve_relevant_content", "retrieve_pdf"):
            break
        if decision.action == "search_papers":
            state.observations.append("OpenAlex search already completed for this request")
            break
        elif decision.action in ("get_paper_details", "retrieve_pdf", "extract_pdf_text", "index_document"):
            if decision.action == "get_paper_details":
                try:
                    paper = state.papers[int(decision.arguments.get("paper_index", 0))]
                except (ValueError, TypeError, IndexError):
                    state.observations.append("No selected paper is available for the requested action")
                    break
                result = tools.get_paper_details(paper, request_id=request_id)
                state.observations.append(f"Selected metadata for {result.get('title') or 'paper'}")
            else:
                _status(state, "Retrieving and indexing selected open PDF", decision.action)
                raw_indices = decision.arguments.get("paper_indices") if decision.action == "retrieve_pdf" else None
                if not isinstance(raw_indices, list) or not raw_indices:
                    raw_indices = [decision.arguments.get("paper_index", 0)]
                selected_indices = []
                for raw_index in raw_indices:
                    try:
                        index = int(raw_index)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= index < len(state.papers) and index not in selected_indices:
                        selected_indices.append(index)
                    if len(selected_indices) == 3:
                        break
                if not selected_indices:
                    state.observations.append("No selected paper is available for the requested action")
                    break
                selected_papers = [state.papers[index] for index in selected_indices]
                results = tools.retrieve_papers(
                    selected_papers,
                    request_id=request_id,
                    query_embedding=query_embedding,
                )
                for result in results:
                    diagnostics["pdfs_downloaded"] += int(result.get("pdf_downloaded", False))
                    diagnostics["pdfs_parsed"] += int(result.get("pdf_parsed", False))
                    diagnostics["chunks_created"] += result.get("chunks_created", 0)
                    diagnostics["embeddings_generated"] += result.get("embeddings_generated", 0)
                    diagnostics["chunks_stored"] += result.get("chunks_stored", 0)
                    diagnostics["paper_cache_hits"] += int(result.get("cache_hit", False))
                selected_chunks = []
                for result in results:
                    selected_chunks.extend(result.get("retrieved_chunks", [])[:2])
                if selected_chunks:
                    state.chunks = selected_chunks + [
                        chunk for chunk in state.chunks if chunk not in selected_chunks
                    ]
                    diagnostics["chunks_retrieved"] = len(state.chunks)
                successes = sum(bool(result.get("success")) for result in results)
                errors = [result["error"] for result in results if result.get("error")]
                observation = f"Processed or reused {successes} of {len(results)} selected papers"
                if errors:
                    observation += ": " + "; ".join(errors)
                state.observations.append(observation)
            state.used_tools.append(decision.action)
            if decision.action == "retrieve_pdf":
                # PDF ingestion already retrieves evidence using the query
                # embedding computed at the start, so another agent turn is not needed.
                break
            _status(state, "Checking paper metadata", decision.action)
        elif decision.action == "retrieve_relevant_content":
            state.observations.append("Local semantic retrieval already completed for this request")
            break

    _status(state, "Synthesizing answer")
    log_process_rss("before_final_synthesis", request_id)
    source_rows = []
    for idx, paper in enumerate(state.papers, 1):
        authors = paper.get("authors") or []
        source_rows.append({"id": f"S{idx}", "title": paper.get("title"),
                            "authors": authors, "year": paper.get("year"), "url": paper.get("url")})
    evidence = "\n\n".join(f"[S{i + 1}] {chunk}" for i, chunk in enumerate(state.chunks[:6]))
    logger.info(
        "[DIAG][%s] Evidence chunks passed to Gemini=%d",
        request_id,
        min(len(state.chunks), 6),
    )
    if not evidence:
        evidence = "No relevant full-text excerpts were available in the local research index."
    sources_text = "\n".join(
        f"[{s['id']}] {s['title']} | {', '.join(s['authors']) if isinstance(s['authors'], list) else s['authors']} ({s['year'] or 'year unknown'}) | {s['url']}"
        for s in source_rows
    ) or "No paper metadata was retrieved."
    try:
        logger.info("[DIAG][%s] Gemini synthesis request model=%s", request_id, "gemini-3.8-flash")
        with measure_stage("Final LLM generation", request_id):
            response = generate_content(
                client,
                request_id=request_id,
                model="gemini-3.8-flash",
                contents=SYNTHESIS_PROMPT.format(query=query, evidence=evidence[:18000], sources=sources_text),
            )
        answer = (response.text or "").strip()
        logger.info(
            "[DIAG][%s] Gemini synthesis response status=success text_present=%s",
            request_id,
            bool(answer),
        )
        if not answer:
            answer = "The retrieved sources do not provide sufficient evidence to answer confidently."
    except Exception as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        logger.error(
            "[DIAG][%s] Gemini synthesis response status=error model=%s code=%s error_type=%s error=%s",
            request_id,
            "gemini-3.8-flash",
            status,
            type(exc).__name__,
            _safe_error_message(exc),
        )
        answer = "The research evidence was collected, but answer synthesis is temporarily unavailable. Please try again."
    _status(state, "Research complete")
    logger.info(
        "[DIAG][%s] Request totals papers=%d pdfs_downloaded=%d pdfs_parsed=%d "
        "chunks_created=%d embeddings_generated=%d chunks_stored=%d "
        "chunks_retrieved=%d chunks_passed_to_gemini=%d paper_cache_hits=%d",
        request_id,
        diagnostics["papers_returned"],
        diagnostics["pdfs_downloaded"],
        diagnostics["pdfs_parsed"],
        diagnostics["chunks_created"],
        diagnostics["embeddings_generated"],
        diagnostics["chunks_stored"],
        diagnostics["chunks_retrieved"],
        min(len(state.chunks), 6),
        diagnostics["paper_cache_hits"],
    )
    logger.info(
        "[TIMING][%s] stage_complete name=Total request duration_seconds=%.3f",
        request_id,
        time.perf_counter() - request_started,
    )
    log_process_rss("request_completed", request_id)
    logger.info("[AGENT][%s] Research completed", request_id)
    return {"success": True, "query": query, "answer": answer,
            "papers": state.papers, "sources": source_rows,
            "retrieved_chunks": state.chunks, "agent_actions": state.actions,
            "research_plan": state.plan, "iterations": state.iteration}
