from services.pdf_parser import extract_text_from_pdf
from services.text_chunker import chunk_text


pdf_path = "data/papers/sample.pdf"


# Step 1: Extract text from PDF
text = extract_text_from_pdf(pdf_path)


# Step 2: Split text into chunks
chunks = chunk_text(text)


print("Total characters:", len(text))
print("Total chunks:", len(chunks))


# Display first 3 chunks
for i, chunk in enumerate(chunks[:3]):

    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)

    print(chunk)