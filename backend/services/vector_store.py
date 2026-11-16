import chromadb
import hashlib


client = chromadb.PersistentClient(
    path="data/chroma"
)


collection = client.get_or_create_collection(
    name="research_papers"
)


def store_chunks(chunks, embeddings, source="local document"):

    ids = [hashlib.sha256(f"{source}:{i}:{chunk[:80]}".encode()).hexdigest()
           for i, chunk in enumerate(chunks)]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": source} for _ in chunks],
    )


def search_chunks(query_embedding, top_k=5):

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results
