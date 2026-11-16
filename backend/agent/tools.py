"""Thin tool adapters around the existing OpenAlex and Chroma RAG services."""
import os
import tempfile
import requests

from services.paper_search import search_papers as _search_papers
from services.embedding_service import create_embeddings
from services.vector_store import search_chunks, store_chunks
from services.pdf_parser import extract_text_from_pdf
from services.text_chunker import chunk_text


def search_papers(query: str, limit: int = 5) -> dict:
    return _search_papers(query, limit=limit)


def retrieve_relevant_content(query: str, top_k: int = 5) -> dict:
    embedding = create_embeddings([query])[0]
    result = search_chunks(embedding, top_k=top_k)
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    chunks = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        chunks.append(f"Source: {metadata.get('source') or 'local research database'}\nExcerpt: {document}")
    return {"chunks": chunks}


def get_paper_details(paper: dict) -> dict:
    """Return selected OpenAlex metadata already supplied by the search result."""
    return {key: paper.get(key) for key in
            ("openalex_id", "title", "authors", "year", "abstract", "url", "pdf_url")}


def retrieve_pdf(paper: dict) -> dict:
    """Download, extract, and index an open PDF; report cleanly when unavailable."""
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return {"success": False, "error": "No open PDF URL is available."}
    path = None
    try:
        response = requests.get(pdf_url, timeout=25, headers={"User-Agent": "AcademicResearchAgent/1.0"})
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            return {"success": False, "error": "The publisher URL did not return a valid PDF."}
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf:
            pdf.write(response.content)
            path = pdf.name
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        if not chunks:
            return {"success": False, "error": "The PDF contained no extractable text."}
        vectors = create_embeddings(chunks)
        store_chunks(chunks, vectors, source=paper.get("title", "Research paper"))
        return {"success": True, "chunks_indexed": len(chunks)}
    except Exception as exc:
        return {"success": False, "error": f"PDF ingestion failed ({type(exc).__name__})."}
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
