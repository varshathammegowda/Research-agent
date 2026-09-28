from services.embedding_service import create_embeddings
from services.vector_store import search_chunks


# --------------------------------------------------
# Question from the user
# --------------------------------------------------

query = "What machine learning techniques are used for IoT security?"


# --------------------------------------------------
# Convert the question into an embedding
# --------------------------------------------------

query_embedding = create_embeddings([query])[0]

print("Query embedding created.")


# --------------------------------------------------
# Search ChromaDB
# --------------------------------------------------

results = search_chunks(
    query_embedding,
    top_k=5
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nTop 5 relevant chunks:\n")


documents = results["documents"][0]

distances = results["distances"][0]


for i, (document, distance) in enumerate(
    zip(documents, distances)
):

    print("=" * 70)

    print(f"RESULT {i + 1}")

    print("=" * 70)

    print(f"Distance: {distance}")

    print("\nText:")

    print(document[:1500])

    print()