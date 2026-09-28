from services.pdf_parser import extract_text_from_pdf
from services.text_chunker import chunk_text
from services.embedding_service import create_embeddings
from services.vector_store import store_chunks


# PDF location
pdf_path = "data/papers/sample.pdf"


# Extract text
text = extract_text_from_pdf(pdf_path)

print("PDF text extracted.")


# Create chunks
chunks = chunk_text(text)

print("Total chunks:", len(chunks))


# Create embeddings
embeddings = create_embeddings(chunks)

print("Embeddings created.")
print("Embedding shape:", embeddings.shape)


# Store in ChromaDB
store_chunks(
    chunks,
    embeddings
)

print("Chunks successfully stored in ChromaDB.")