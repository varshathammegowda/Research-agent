"""Persistent ChromaDB access for indexed research papers and local documents."""

import hashlib
import os
from pathlib import Path

import chromadb

from services.timing import log_process_rss, measure_stage


CHROMA_PATH = Path(
    os.getenv("CHROMA_DATA_PATH", str(Path(__file__).resolve().parents[1] / "data" / "chroma"))
).resolve()
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# Keep one persistent client and collection for the lifetime of this process.
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="research_papers")
processed_papers = client.get_or_create_collection(name="processed_papers")
log_process_rss("after_chroma_initialization")


def is_paper_indexed(paper_id, legacy_source=None, request_id="unknown"):
    """Check the persistent collection before downloading or embedding a PDF.

    The source-title fallback lets indexes created before stable paper IDs were
    added remain reusable, while new entries are keyed by OpenAlex ID or DOI.
    """
    with measure_stage("ChromaDB paper cache lookup", request_id):
        if paper_id:
            manifest_id = hashlib.sha256(paper_id.encode()).hexdigest()
            manifest = processed_papers.get(ids=[manifest_id], include=["metadatas"])
            manifest_metadata = (manifest.get("metadatas") or [None])[0]
            if manifest_metadata and manifest_metadata.get("complete"):
                return int(manifest_metadata.get("chunk_count") or 0)

            result = collection.get(where={"paper_id": paper_id}, include=["metadatas"])
            metadata = result.get("metadatas") or []
            # Entries produced before manifests were introduced remain cacheable.
            # New batched writes are marked incomplete until their manifest commits.
            if result.get("ids") and any(item.get("cacheable", True) for item in metadata):
                return len(result["ids"])

        if legacy_source:
            result = collection.get(where={"source": legacy_source}, include=["metadatas"])
            metadata = result.get("metadatas") or []
            return sum(
                1 for item in metadata
                if not item.get("paper_id") and item.get("cacheable", True)
            )

    return 0


def store_chunks(
    chunks,
    embeddings,
    source="local document",
    paper_id=None,
    request_id="unknown",
    chunk_offset=0,
    cacheable=True,
):
    with measure_stage("ChromaDB upsert", request_id):
        identity = paper_id or source
        ids = [
            hashlib.sha256(f"{identity}:{chunk_offset + i}:{chunk}".encode()).hexdigest()
            for i, chunk in enumerate(chunks)
        ]
        metadatas = [
            {"source": source, "cacheable": cacheable, **({"paper_id": paper_id} if paper_id else {})}
            for _ in chunks
        ]
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )


def mark_paper_processed(paper_id, source, chunk_count, request_id="unknown"):
    """Commit a completion marker after all bounded embedding batches are stored."""
    with measure_stage("ChromaDB paper completion marker", request_id):
        manifest_id = hashlib.sha256(paper_id.encode()).hexdigest()
        processed_papers.upsert(
            ids=[manifest_id],
            documents=[source or paper_id],
            embeddings=[[0.0] * 384],
            metadatas=[{
                "paper_id": paper_id,
                "source": source,
                "chunk_count": int(chunk_count),
                "complete": True,
            }],
        )


def clear_incomplete_paper(paper_id, request_id="unknown"):
    """Remove orphaned batch rows left by a failed, uncommitted ingestion."""
    if not paper_id:
        return
    with measure_stage("ChromaDB incomplete paper cleanup", request_id):
        result = collection.get(where={"paper_id": paper_id}, include=["metadatas"])
        metadatas = result.get("metadatas") or []
        ids = result.get("ids") or []
        if ids and not any(item.get("cacheable", True) for item in metadatas):
            collection.delete(ids=ids)


def search_chunks(query_embedding, top_k=5, request_id="unknown", paper_id=None, source=None):
    with measure_stage("ChromaDB similarity search", request_id):
        query = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
        }
        if paper_id:
            query["where"] = {"paper_id": paper_id}
        elif source:
            query["where"] = {"source": source}
        return collection.query(**query)
