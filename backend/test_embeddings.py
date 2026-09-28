from services.pdf_parser import extract_text_from_pdf
from services.text_chunker import chunk_text
from services.embedding_service import create_embeddings


# PDF location
pdf_path = "data/papers/sample.pdf"


# Step 1: Extract PDF text
text = extract_text_from_pdf(pdf_path)


# Step 2: Create chunks
chunks = chunk_text(text)


print("Total chunks:", len(chunks))


# Step 3: Create embeddings
embeddings = create_embeddings(chunks)


print("Embedding shape:", embeddings.shape)


# Display first embedding
print("\nFirst chunk embedding:")
print(embeddings[0])