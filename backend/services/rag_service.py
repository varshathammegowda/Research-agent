"""Single-call RAG answer generation over retrieved research evidence."""

import os
import time

from dotenv import load_dotenv
from google import genai

from services.embedding_service import create_embeddings
from services.gemini_generation import generate_content
from services.timing import log_process_rss, measure_stage
from services.vector_store import search_chunks

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client(api_key=api_key)


def generate_answer(
    query: str,
    top_k: int = 5,
    request_id: str = "unknown",
    query_embedding=None,
    retrieval_result=None,
    papers=None,
):
    """Retrieve up to top_k chunks when needed, then make one synthesis call."""
    papers = papers or []
    if retrieval_result is None:
        if query_embedding is None:
            query_embedding = create_embeddings([query], request_id=request_id)[0]
        retrieval_result = search_chunks(
            query_embedding,
            top_k=top_k,
            request_id=request_id,
        )
        retrieved_chunks = (retrieval_result.get("documents") or [[]])[0]
    else:
        retrieved_chunks = (retrieval_result.get("chunks") or [])[:top_k]

    sources = []
    abstract_evidence = []
    for index, paper in enumerate(papers, 1):
        source_id = f"S{index}"
        title = paper.get("title") or "Untitled paper"
        authors = paper.get("authors") or []
        sources.append({
            "id": source_id,
            "title": title,
            "authors": authors,
            "year": paper.get("year"),
            "url": paper.get("url"),
        })
        abstract = (paper.get("abstract") or "").strip()
        if abstract:
            abstract_evidence.append(f"[{source_id}] {title}\nAbstract: {abstract[:3000]}")

    evidence = "\n\n".join(retrieved_chunks)
    abstract_context = "\n\n".join(abstract_evidence)
    if not evidence:
        evidence = "No full-text excerpts were available in the local research index."
    if not abstract_context:
        abstract_context = "No OpenAlex abstracts were available."
    sources_context = "\n".join(
        f"[{source['id']}] {source['title']} | "
        f"{', '.join(source['authors']) if isinstance(source['authors'], list) else source['authors']} "
        f"({source['year'] or 'year unknown'}) | {source['url'] or 'URL unavailable'}"
        for source in sources
    ) or "No OpenAlex paper metadata was retrieved."

    log_process_rss(
        "before_gemini_context_construction",
        request_id,
        context_type="final_synthesis",
        chunks=len(retrieved_chunks),
        papers=len(papers),
    )
    prompt = f"""You are an academic research assistant. Answer the question using the supplied retrieved excerpts and OpenAlex abstracts. Cite claims with the source labels shown in the context (for example [S1] or the source label attached to a retrieved excerpt). Do not invent findings, papers, authors, dates, or citations. If the evidence is insufficient, say so clearly. Keep the answer concise and academic.

Question:
{query}

Top retrieved full-text excerpts:
{evidence}

OpenAlex abstracts:
{abstract_context}

Paper metadata:
{sources_context}
"""
    log_process_rss(
        "after_gemini_context_construction",
        request_id,
        context_type="final_synthesis",
        evidence_chars=len(evidence),
        abstract_chars=len(abstract_context),
        sources_chars=len(sources_context),
        prompt_chars=len(prompt),
    )

    request_started = time.perf_counter()
    response = None
    response_status = "success"
    try:
        log_process_rss(
            "before_final_gemini_call",
            request_id,
            prompt_chars=len(prompt),
        )
        with measure_stage("Final LLM generation", request_id):
            response = generate_content(
                client,
                request_id=request_id,
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            log_process_rss(
                "after_final_gemini_call",
                request_id,
                elapsed_seconds=f"{time.perf_counter() - request_started:.3f}",
                response_chars=len(response.text or ""),
            )
        answer = (response.text or "").strip()
        if not answer:
            answer = "The retrieved sources do not provide sufficient evidence to answer confidently."
    except Exception:
        response_status = "error"
        raise
    finally:
        log_process_rss(
            "after_gemini_response_processed",
            request_id,
            response_status=response_status,
            response_chars=len(response.text or "") if response is not None else 0,
            answer_chars=len(answer) if "answer" in locals() else 0,
        )
        log_process_rss(
            "after_final_synthesis",
            request_id,
            elapsed_seconds=f"{time.perf_counter() - request_started:.3f}",
        )

    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "paper_sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }
