"""Thin tool adapters around the existing OpenAlex and Chroma RAG services."""
import os
import tempfile
import logging
import requests

from services.paper_search import search_papers as _search_papers
from services.embedding_service import create_embeddings
from services.vector_store import search_chunks, store_chunks
from services.pdf_parser import extract_text_from_pdf
from services.text_chunker import chunk_text

logger = logging.getLogger(__name__)


def search_papers(query: str, limit: int = 5, request_id: str = "unknown") -> dict:
    result = _search_papers(query, limit=limit)
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


def retrieve_relevant_content(query: str, top_k: int = 5, request_id: str = "unknown") -> dict:
    embedding = create_embeddings([query])[0]
    logger.info("[DIAG][%s] Query embeddings generated=1", request_id)
    result = search_chunks(embedding, top_k=top_k)
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    chunks = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        chunks.append(f"Source: {metadata.get('source') or 'local research database'}\nExcerpt: {document}")
    logger.info("[DIAG][%s] ChromaDB chunks_retrieved=%d", request_id, len(chunks))
    return {"chunks": chunks, "query_embeddings_generated": 1}


def get_paper_details(paper: dict) -> dict:
    """Return selected OpenAlex metadata already supplied by the search result."""
    return {key: paper.get(key) for key in
            ("openalex_id", "title", "authors", "year", "abstract", "url", "pdf_url")}


def retrieve_pdf(paper: dict, request_id: str = "unknown") -> dict:
    """Download, extract, and index an open PDF; report cleanly when unavailable."""
    pdf_url = paper.get("pdf_url")
    title = paper.get("title") or ""
    metrics = {
        "pdf_downloaded": False,
        "pdf_parsed": False,
        "chunks_created": 0,
        "embeddings_generated": 0,
        "chunks_stored": 0,
    }
    if not pdf_url:
        logger.info("[DIAG][%s] PDF unavailable title=%r pdf_url_exists=false", request_id, title)
        return {"success": False, "error": "No open PDF URL is available.", **metrics}
    path = None
    stage = "download"
    try:
        response = requests.get(pdf_url, timeout=25, headers={"User-Agent": "AcademicResearchAgent/1.0"})
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            logger.warning("[DIAG][%s] PDF download failed title=%r reason=invalid_pdf_response", request_id, title)
            return {"success": False, "error": "The publisher URL did not return a valid PDF.", **metrics}
        metrics["pdf_downloaded"] = True
        logger.info("[DIAG][%s] PDF downloaded title=%r bytes=%d", request_id, title, len(response.content))
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf:
            pdf.write(response.content)
            path = pdf.name
        stage = "PyMuPDF extraction"
        text = extract_text_from_pdf(path)
        metrics["pdf_parsed"] = True
        logger.info("[DIAG][%s] PyMuPDF parsed title=%r extracted_characters=%d", request_id, title, len(text))
        stage = "chunking"
        chunks = chunk_text(text)
        metrics["chunks_created"] = len(chunks)
        logger.info("[DIAG][%s] Text chunks created title=%r count=%d", request_id, title, len(chunks))
        if not chunks:
            return {"success": False, "error": "The PDF contained no extractable text.", **metrics}
        stage = "embedding generation"
        vectors = create_embeddings(chunks)
        metrics["embeddings_generated"] = len(vectors)
        logger.info("[DIAG][%s] Chunk embeddings generated title=%r count=%d", request_id, title, len(vectors))
        stage = "ChromaDB storage"
        store_chunks(chunks, vectors, source=paper.get("title", "Research paper"))
        metrics["chunks_stored"] = len(chunks)
        logger.info("[DIAG][%s] ChromaDB chunks stored title=%r count=%d", request_id, title, len(chunks))
        return {"success": True, "chunks_indexed": len(chunks), **metrics}
    except Exception as exc:
        logger.error(
            "[DIAG][%s] PDF ingestion failed title=%r stage=%s error_type=%s",
            request_id,
            title,
            stage,
            type(exc).__name__,
        )
        return {"success": False, "error": f"PDF ingestion failed ({type(exc).__name__}).", **metrics}
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
