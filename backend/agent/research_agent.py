"""Simple, bounded tool-using academic research agent."""
import json
import logging
import os
import re
import uuid

from pydantic import ValidationError

from .prompts import DECISION_PROMPT, SYNTHESIS_PROMPT
from .schemas import AgentDecision, ResearchState
from . import tools

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 5


def _status(state: ResearchState, message: str, tool: str = "") -> None:
    state.actions.append({"status": message, "tool": tool})


def _decision(client, state: ResearchState, request_id: str = "unknown") -> AgentDecision:
    prompt = DECISION_PROMPT.format(
        query=state.query, objective=state.objective or state.query,
        plan=json.dumps(state.plan), used=json.dumps(state.used_tools),
        paper_count=len(state.papers), chunk_count=len(state.chunks),
        observations=json.dumps(state.observations[-4:]),
    )
    logger.info("[DIAG][%s] Gemini decision request model=%s", request_id, "gemini-3.8-flash")
    response = client.models.generate_content(
        model="gemini-3.8-flash", contents=prompt,
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
    request_id = uuid.uuid4().hex[:8]
    diagnostics = {
        "papers_returned": 0,
        "pdfs_downloaded": 0,
        "pdfs_parsed": 0,
        "chunks_created": 0,
        "embeddings_generated": 0,
        "chunks_stored": 0,
        "chunks_retrieved": 0,
    }
    state = ResearchState(query=query)
    state.objective = query.strip()
    state.plan = ["Find relevant papers", "Retrieve available research evidence", "Synthesize a cited answer"]
    _status(state, "Planning research")
    logger.info("[AGENT][%s] Research started", request_id)
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
            term = decision.arguments.get("query") or query
            _status(state, "Searching OpenAlex", "search_papers")
            state.used_tools.append("search_papers")
            try:
                result = tools.search_papers(str(term), limit=5, request_id=request_id)
                if result.get("success"):
                    state.papers = result.get("papers", [])
                    diagnostics["papers_returned"] = len(state.papers)
                    state.observations.append(f"OpenAlex returned {len(state.papers)} papers")
                else:
                    state.observations.append("OpenAlex search unavailable")
            except Exception as exc:
                logger.exception("[TOOL] OpenAlex failed")
                state.observations.append(f"OpenAlex search failed: {type(exc).__name__}")
        elif decision.action in ("get_paper_details", "retrieve_pdf", "extract_pdf_text", "index_document"):
            index = decision.arguments.get("paper_index", 0)
            try:
                paper = state.papers[int(index)]
            except (ValueError, TypeError, IndexError):
                state.observations.append("No selected paper is available for the requested action")
                continue
            if decision.action == "get_paper_details":
                result = tools.get_paper_details(paper)
                state.observations.append(f"Selected metadata for {result.get('title') or 'paper'}")
            else:
                result = tools.retrieve_pdf(paper, request_id=request_id)
                diagnostics["pdfs_downloaded"] += int(result.get("pdf_downloaded", False))
                diagnostics["pdfs_parsed"] += int(result.get("pdf_parsed", False))
                diagnostics["chunks_created"] += result.get("chunks_created", 0)
                diagnostics["embeddings_generated"] += result.get("embeddings_generated", 0)
                diagnostics["chunks_stored"] += result.get("chunks_stored", 0)
                state.observations.append(result.get("error") or f"Indexed {result.get('chunks_indexed', 0)} PDF chunks")
            state.used_tools.append(decision.action)
            _status(state, "Retrieving and indexing open paper PDF" if decision.action == "retrieve_pdf" else "Checking paper metadata", decision.action)
        elif decision.action == "retrieve_relevant_content":
            term = decision.arguments.get("query") or query
            _status(state, "Retrieving semantic research evidence", "retrieve_relevant_content")
            state.used_tools.append("retrieve_relevant_content")
            try:
                retrieval = tools.retrieve_relevant_content(
                    str(term), request_id=request_id
                )
                state.chunks = retrieval.get("chunks", [])
                diagnostics["embeddings_generated"] += retrieval.get(
                    "query_embeddings_generated", 0
                )
                diagnostics["chunks_retrieved"] = len(state.chunks)
                state.observations.append(f"Retrieved {len(state.chunks)} evidence chunks")
            except Exception as exc:
                logger.exception("[TOOL] Semantic retrieval failed")
                state.observations.append(f"Semantic retrieval failed: {type(exc).__name__}")

    _status(state, "Synthesizing answer")
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
        response = client.models.generate_content(
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
        "chunks_retrieved=%d chunks_passed_to_gemini=%d",
        request_id,
        diagnostics["papers_returned"],
        diagnostics["pdfs_downloaded"],
        diagnostics["pdfs_parsed"],
        diagnostics["chunks_created"],
        diagnostics["embeddings_generated"],
        diagnostics["chunks_stored"],
        diagnostics["chunks_retrieved"],
        min(len(state.chunks), 6),
    )
    logger.info("[AGENT][%s] Research completed", request_id)
    return {"success": True, "query": query, "answer": answer,
            "papers": state.papers, "sources": source_rows,
            "retrieved_chunks": state.chunks, "agent_actions": state.actions,
            "research_plan": state.plan, "iterations": state.iteration}
