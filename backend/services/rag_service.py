import os

from dotenv import load_dotenv
from google import genai

from services.embedding_service import create_embeddings
from services.vector_store import search_chunks


# Load environment variables
load_dotenv()


# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


# Create Gemini client
client = genai.Client(
    api_key=api_key
)


def generate_answer(query: str, top_k: int = 5):

    # --------------------------------------------------
    # 1. Convert user question into an embedding
    # --------------------------------------------------

    query_embedding = create_embeddings(
        [query]
    )[0]


    # --------------------------------------------------
    # 2. Retrieve relevant chunks from ChromaDB
    # --------------------------------------------------

    results = search_chunks(
        query_embedding,
        top_k=top_k
    )


    documents = results["documents"][0]


    # --------------------------------------------------
    # 3. Combine retrieved chunks into context
    # --------------------------------------------------

    context = "\n\n".join(
        documents
    )


    # --------------------------------------------------
    # 4. Create RAG prompt
    # --------------------------------------------------

    prompt = f"""
You are an academic research assistant.

Answer the user's question using ONLY the
research context provided below.

If the answer is not present in the context,
say that the information is not available
in the provided research paper.

Research Context:
-----------------
{context}
-----------------

User Question:
{query}

Give a clear and concise academic answer.
"""


    # --------------------------------------------------
    # 5. Send context + question to Gemini
    # --------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )


    # --------------------------------------------------
    # 6. Return answer and retrieved chunks
    # --------------------------------------------------

    return {
        "answer": response.text,
        "sources": documents
    }