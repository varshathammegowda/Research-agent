"""Paper search, PDF processing, and retrieval helpers for the fixed RAG pipeline."""
import os
import tempfile
import logging
import hashlib
import time
import requests
from threading import BoundedSemaphore, Lock
from concurrent.futures import ThreadPoolExecutor

from services.paper_search import MAX_OPENALEX_PAPERS, search_papers as _search_papers
from services.embedding_service import create_embeddings
from services.vector_store import (
    clear_incomplete_paper,
    is_paper_indexed,
    mark_paper_processed,
    search_chunks,
    store_chunks,
)
from services.pdf_parser import extract_text_from_pdf
from services.text_chunker import chunk_text
from services.timing import log_process_rss, measure_stage

logger = logging.getLogger(__name__)
_PAPER_LOCKS = tuple(Lock() for _ in range(64))
MAX_CONCURRENT_PAPER_PROCESSING = 2
PAPER_EMBED_BATCH_SIZE = 32
_PAPER_PROCESSING_LIMIT = BoundedSemaphore(MAX_CONCURRENT_PAPER_PROCESSING)


def search_papers(
    query: str,
    limit: int = MAX_OPENALEX_PAPERS,
    request_id: str = "unknown",
) -> dict:
    limit = max(1, min(int(limit), MAX_OPENALEX_PAPERS))
    with measure_stage("OpenAlex search and paper metadata", request_id):
        result = _search_papers(query, limit=limit, request_id=request_id)
    papers = result.get("papers", []) if result.get("success") else []
    logger.info(
        "[DIAG][%s] OpenAlex status=%s papers_returned=%d",
        request_id,
        "success" if result.get("success") else "error",
        len(papers),
    )
    for index, paper in enumerate(papers, 1):
        logger.info(
            "[DIAG][%s] OpenAlex paper=%d title=%r pdf_url_exists=%s",
            request_id,
            index,
            paper.get("title") or "",
            bool(paper.get("pdf_url")),
        )
    return result


def retrieve_relevant_content(
    query: str,
    top_k: int = 5,
    request_id: str = "unknown",
    query_embedding=None,
    papers=None,
) -> dict:
    embedding = query_embedding
    embeddings_generated = 0
    if embedding is None:
        embedding = create_embeddings([query], request_id=request_id)[0]
        embeddings_generated = 1
    logger.info("[DIAG][%s] Query embeddings generated=%d", request_id, embeddings_generated)
    paper_ids = None
    sources = None
    if papers is not None:
        papers = papers[:MAX_OPENALEX_PAPERS]
        if not papers:
            return {
                "chunks": [],
                "query_embeddings_generated": embeddings_generated,
                "query_embedding": embedding,
            }
        paper_ids = [_paper_identifier(paper) for paper in papers]
        sources = [paper.get("title") for paper in papers if paper.get("title")]
    try:
        result = search_chunks(
            embedding,
            top_k=top_k,
            request_id=request_id,
            paper_ids=paper_ids,
            sources=sources,
        )
    except Exception as exc:
        logger.error(
            "[DIAG][%s] ChromaDB retrieval failed error_type=%s",
            request_id,
            type(exc).__name__,
        )
        return {
            "chunks": [],
            "query_embeddings_generated": embeddings_generated,
            "query_embedding": embedding,
            "error": f"ChromaDB retrieval failed ({type(exc).__name__}).",
        }
    chunks = _format_chunks(result)
    logger.info("[DIAG][%s] ChromaDB chunks_retrieved=%d", request_id, len(chunks))
    return {
        "chunks": chunks,
        "query_embeddings_generated": embeddings_generated,
        "query_embedding": embedding,
    }


def _format_chunks(result: dict) -> list[str]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    return [
        f"Source: {(metadatas[index] if index < len(metadatas) else {}).get('source') or 'local research database'}\nExcerpt: {document}"
        for index, document in enumerate(documents)
    ]


def get_paper_details(paper: dict, request_id: str = "unknown") -> dict:
    """Return selected OpenAlex metadata already supplied by the search result."""
    with measure_stage("Paper metadata retrieval", request_id):
        return {key: paper.get(key) for key in
                ("openalex_id", "title", "authors", "year", "abstract", "url", "pdf_url")}


def _paper_identifier(paper: dict) -> str:
    """Return a stable cache key for OpenAlex works, DOI records, or PDF URLs."""
    stable_id = paper.get("openalex_id") or paper.get("doi") or paper.get("pdf_url")
    if stable_id:
        return str(stable_id).strip().lower()
    fallback = " ".join((paper.get("title") or "").casefold().split())
    return "title:" + hashlib.sha256(fallback.encode()).hexdigest()


def retrieve_pdf(paper: dict, request_id: str = "unknown", query_embedding=None) -> dict:
    paper_id = _paper_identifier(paper)
    lock_slot = int(hashlib.sha256(paper_id.encode()).hexdigest()[:4], 16) % len(_PAPER_LOCKS)
    with measure_stage("PDF processing lock wait", request_id):
        _PAPER_LOCKS[lock_slot].acquire()
    try:
        return _retrieve_pdf_locked(paper, paper_id, request_id, query_embedding)
    finally:
        _PAPER_LOCKS[lock_slot].release()


def retrieve_papers(papers: list[dict], request_id: str = "unknown", query_embedding=None) -> list[dict]:
    """Process only the deterministic paper selection, preserving input order."""
    papers = papers[:MAX_OPENALEX_PAPERS]
    logger.info(
        "[DIAG][%s] selected_papers=%d configured_paper_processing_concurrency=%d",
        request_id,
        len(papers),
        MAX_CONCURRENT_PAPER_PROCESSING,
    )

    def process(index, paper):
        started = time.perf_counter()
        try:
            paper_id_short = hashlib.sha256(_paper_identifier(paper).encode()).hexdigest()[:10]
        except Exception:
            paper_id_short = "unavailable"
        logger.info(
            "[DIAG][%s] paper_processing_start selected_index=%d paper_id=%s",
            request_id,
            index,
            paper_id_short,
        )
        log_process_rss("before_paper_processing", request_id)
        try:
            return retrieve_pdf(paper, request_id=request_id, query_embedding=query_embedding)
        except Exception as exc:
            logger.error(
                "[DIAG][%s] Paper processing failed paper_id=%s error_type=%s",
                request_id,
                paper_id_short,
                type(exc).__name__,
            )
            return {"success": False, "error": f"Paper processing failed ({type(exc).__name__})."}
        finally:
            logger.info(
                "[DIAG][%s] paper_processing_finish selected_index=%d paper_id=%s duration_seconds=%.3f",
                request_id,
                index,
                paper_id_short,
                time.perf_counter() - started,
            )
            log_process_rss("after_paper_processing", request_id)

    with measure_stage("Total selected paper processing", request_id):
        if len(papers) <= 1:
            return [process(index, paper) for index, paper in enumerate(papers)]
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_PAPER_PROCESSING, len(papers))) as executor:
            futures = [executor.submit(process, index, paper) for index, paper in enumerate(papers)]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error(
                        "[DIAG][%s] Paper worker failed error_type=%s",
                        request_id,
                        type(exc).__name__,
                    )
                    results.append({"success": False, "error": f"Paper processing failed ({type(exc).__name__})."})
            return results


def _retrieve_pdf_locked(paper: dict, paper_id: str, request_id: str, query_embedding) -> dict:
    """Download, extract, and index an open PDF; report cleanly when unavailable."""
    paper_started = time.perf_counter()
    safe_paper_id = hashlib.sha256(paper_id.encode()).hexdigest()[:10]
    log_process_rss(
        "paper_start",
        request_id,
        paper_id=safe_paper_id,
        elapsed_seconds="0.000",
    )
    pdf_url = paper.get("pdf_url")
    title = paper.get("title") or ""
    metrics = {
        "pdf_downloaded": False,
        "pdf_parsed": False,
        "chunks_created": 0,
        "embeddings_generated": 0,
        "chunks_stored": 0,
        "cache_hit": False,
    }
    cached_chunks = is_paper_indexed(paper_id, legacy_source=title, request_id=request_id)
    if cached_chunks:
        metrics["cache_hit"] = True
        logger.info(
            "[DIAG][%s] Reusing indexed paper paper_id=%s chunks=%d",
            request_id,
            paper_id,
            cached_chunks,
        )
        result = {"success": True, "cached": True, "chunks_indexed": cached_chunks, **metrics}
        if query_embedding is not None:
            retrieval = search_chunks(
                query_embedding,
                top_k=5,
                request_id=request_id,
                paper_id=paper_id,
            )
            result["retrieved_chunks"] = _format_chunks(retrieval)
            if not result["retrieved_chunks"]:
                # Legacy title-keyed chunks predate paper_id metadata.
                result["retrieved_chunks"] = _format_chunks(
                    search_chunks(query_embedding, top_k=5, request_id=request_id, source=title)
                )
        log_process_rss(
            "paper_cleanup",
            request_id,
            paper_id=safe_paper_id,
            elapsed_seconds=f"{time.perf_counter() - paper_started:.3f}",
        )
        return result
    clear_incomplete_paper(paper_id, request_id=request_id)
    if not pdf_url:
        logger.info("[DIAG][%s] PDF unavailable title=%r pdf_url_exists=false", request_id, title)
        log_process_rss(
            "paper_cleanup",
            request_id,
            paper_id=safe_paper_id,
            elapsed_seconds=f"{time.perf_counter() - paper_started:.3f}",
        )
        return {"success": False, "error": "No open PDF URL is available.", **metrics}
    with measure_stage("PDF memory slot wait", request_id):
        _PAPER_PROCESSING_LIMIT.acquire()

    path = None
    response = None
    text = None
    chunks = None
    batch_chunks = None
    batch_vectors = None
    retrieval = None
    stage = "download"
    try:
        with measure_stage("PDF download", request_id):
            response = requests.get(
                pdf_url,
                timeout=25,
                headers={"User-Agent": "AcademicResearchAgent/1.0"},
                stream=True,
            )
            response.raise_for_status()
            pdf_bytes = 0
            pdf_header = bytearray()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf:
                path = pdf.name
                for block in response.iter_content(chunk_size=64 * 1024):
                    if not block:
                        continue
                    if len(pdf_header) < 4:
                        pdf_header.extend(block[: 4 - len(pdf_header)])
                    pdf.write(block)
                    pdf_bytes += len(block)
                block = None
        response.close()
        response = None
        log_process_rss(
            "after_pdf_download",
            request_id,
            paper_id=safe_paper_id,
            elapsed_seconds=f"{time.perf_counter() - paper_started:.3f}",
        )
        if bytes(pdf_header) != b"%PDF":
            logger.warning("[DIAG][%s] PDF download failed reason=invalid_pdf_response", request_id)
            return {"success": False, "error": "The publisher URL did not return a valid PDF.", **metrics}
        metrics["pdf_downloaded"] = True
        logger.info("[DIAG][%s] PDF downloaded bytes=%d", request_id, pdf_bytes)

        stage = "PyMuPDF extraction"
        text = extract_text_from_pdf(path, request_id=request_id)
        metrics["pdf_parsed"] = True
        logger.info("[DIAG][%s] PyMuPDF parsed extracted_characters=%d", request_id, len(text))
        log_process_rss(
            "after_pdf_extraction",
            request_id,
            paper_id=safe_paper_id,
            elapsed_seconds=f"{time.perf_counter() - paper_started:.3f}",
        )

        stage = "chunking"
        chunks = chunk_text(text, request_id=request_id)
        text = None
        metrics["chunks_created"] = len(chunks)
        logger.info("[DIAG][%s] Text chunks created count=%d", request_id, len(chunks))
        if not chunks:
            return {"success": False, "error": "The PDF contained no extractable text.", **metrics}

        stage = "embedding generation and ChromaDB storage"
        log_process_rss("before_paper_embedding", request_id)
        source = paper.get("title") or "Research paper"
        chunk_offset = 0
        batch_index = 0
        while chunks:
            batch_chunks = chunks[:PAPER_EMBED_BATCH_SIZE]
            del chunks[:PAPER_EMBED_BATCH_SIZE]
            batch_vectors = create_embeddings(
                batch_chunks,
                request_id=request_id,
                diagnostic_fields={"paper_id": safe_paper_id, "batch": batch_index},
            )
            metrics["embeddings_generated"] += len(batch_vectors)
            store_chunks(
                batch_chunks,
                batch_vectors,
                source=source,
                paper_id=paper_id,
                request_id=request_id,
                chunk_offset=chunk_offset,
                cacheable=False,
            )
            metrics["chunks_stored"] += len(batch_chunks)
            chunk_offset += len(batch_chunks)
            batch_chunks = None
            batch_vectors = None
            log_process_rss(
                "after_embedding_batch_cleanup",
                request_id,
                paper_id=safe_paper_id,
                batch=batch_index,
            )
            batch_index += 1

        mark_paper_processed(paper_id, source, metrics["chunks_stored"], request_id=request_id)
        logger.info("[DIAG][%s] ChromaDB chunks stored count=%d", request_id, metrics["chunks_stored"])
        chunks = None

        result = {"success": True, "chunks_indexed": metrics["chunks_stored"], **metrics}
        if query_embedding is not None:
            retrieval = search_chunks(
                query_embedding,
                top_k=5,
                request_id=request_id,
                paper_id=paper_id,
            )
            result["retrieved_chunks"] = _format_chunks(retrieval)
        return result
    except Exception as exc:
        logger.error(
            "[DIAG][%s] PDF ingestion failed paper_id=%s stage=%s error_type=%s",
            request_id,
            hashlib.sha256(paper_id.encode()).hexdigest()[:10],
            stage,
            type(exc).__name__,
        )
        return {"success": False, "error": f"PDF ingestion failed ({type(exc).__name__}).", **metrics}
    finally:
        try:
            if response is not None:
                response.close()
        except Exception as cleanup_error:
            logger.warning(
                "[DIAG][%s] PDF response cleanup failed error_type=%s",
                request_id,
                type(cleanup_error).__name__,
            )
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError as cleanup_error:
            logger.warning(
                "[DIAG][%s] Temporary PDF cleanup failed error_type=%s",
                request_id,
                type(cleanup_error).__name__,
            )
        finally:
            text = None
            chunks = None
            batch_chunks = None
            batch_vectors = None
            retrieval = None
            try:
                _PAPER_PROCESSING_LIMIT.release()
            finally:
                log_process_rss(
                    "paper_cleanup",
                    request_id,
                    paper_id=safe_paper_id,
                    elapsed_seconds=f"{time.perf_counter() - paper_started:.3f}",
                )
