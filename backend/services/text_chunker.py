from services.timing import timed_stage

@timed_stage("Text chunking")
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    request_id: str = "unknown",
):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
