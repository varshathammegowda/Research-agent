from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embeddings(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings