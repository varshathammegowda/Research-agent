"""Simple, bounded tool-using academic research agent."""
import json
import logging

from pydantic import ValidationError

from services.query_validator import validate_query, INVALID_QUERY_MESSAGE
from .prompts import DECISION_PROMPT, SYNTHESIS_PROMPT
from .schemas import AgentDecision, ResearchState
from . import tools

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 5


def _status(state: ResearchState, message: str, tool: str = "") -> None:
    state.actions.append({"status": message, "tool": tool})


def _decision(client, state: ResearchState) -> AgentDecision:
    prompt = DECISION_PROMPT.format(
        query=state.query, objective=state.objective or state.query,
        plan=json.dumps(state.plan), used=json.dumps(state.used_tools),
        paper_count=len(state.papers), chunk_count=len(state.chunks),
        observations=json.dumps(state.observations[-4:]),
    )
    response = client.models.generate_content(
        model="gemini-3.8-flash", contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return AgentDecision.model_validate_json(response.text or "")


def run(client, query: str) -> dict:
    state = ResearchState(query=query)
    _status(state, "Validating research question")
    validation = validate_query(query, client)
    if not validation.get("valid"):
        return {"success": False, "query": query,
                "error": ("Query validation is temporarily unavailable. Please try again shortly."
                          if validation.get("error") else INVALID_QUERY_MESSAGE),
                "papers": [], "sources": [], "agent_actions": state.actions}

    state.objective = query.strip()
    state.plan = ["Find relevant papers", "Retrieve available research evidence", "Synthesize a cited answer"]
    _status(state, "Planning research")
    logger.info("[AGENT] Research started")
    for turn in range(MAX_ITERATIONS):
        state.iteration = turn + 1
        try:
            decision = _decision(client, state)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("[AGENT] Invalid decision; using safe fallback: %s", type(exc).__name__)
            decision = AgentDecision(action="finish", reason="Decision could not be validated")
        except Exception as exc:
            logger.exception("[AGENT] Decision model failed")
            state.observations.append(f"Decision model unavailable: {type(exc).__name__}")
            decision = AgentDecision(action="finish")

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
                result = tools.search_papers(str(term), limit=5)
                if result.get("success"):
                    state.papers = result.get("papers", [])
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
                result = tools.retrieve_pdf(paper)
                state.observations.append(result.get("error") or f"Indexed {result.get('chunks_indexed', 0)} PDF chunks")
            state.used_tools.append(decision.action)
            _status(state, "Retrieving and indexing open paper PDF" if decision.action == "retrieve_pdf" else "Checking paper metadata", decision.action)
        elif decision.action == "retrieve_relevant_content":
            term = decision.arguments.get("query") or query
            _status(state, "Retrieving semantic research evidence", "retrieve_relevant_content")
            state.used_tools.append("retrieve_relevant_content")
            try:
                state.chunks = tools.retrieve_relevant_content(str(term)).get("chunks", [])
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
    if not evidence:
        evidence = "No relevant full-text excerpts were available in the local research index."
    sources_text = "\n".join(
        f"[{s['id']}] {s['title']} | {', '.join(s['authors']) if isinstance(s['authors'], list) else s['authors']} ({s['year'] or 'year unknown'}) | {s['url']}"
        for s in source_rows
    ) or "No paper metadata was retrieved."
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=SYNTHESIS_PROMPT.format(query=query, evidence=evidence[:18000], sources=sources_text),
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "The retrieved sources do not provide sufficient evidence to answer confidently."
    except Exception:
        logger.exception("[AGENT] Synthesis failed")
        answer = "The research evidence was collected, but answer synthesis is temporarily unavailable. Please try again."
    _status(state, "Research complete")
    logger.info("[AGENT] Research completed")
    return {"success": True, "query": query, "answer": answer,
            "papers": state.papers, "sources": source_rows,
            "retrieved_chunks": state.chunks, "agent_actions": state.actions,
            "research_plan": state.plan, "iterations": state.iteration}
