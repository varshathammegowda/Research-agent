from services.rag_service import generate_answer


query = "What machine learning techniques are used for IoT security?"


result = generate_answer(query)


print("\n" + "=" * 70)
print("RAG ANSWER")
print("=" * 70)

print(result["answer"])


print("\n" + "=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)


for i, source in enumerate(
    result["sources"],
    start=1
):

    print(f"\nSOURCE {i}")
    print("-" * 50)
    print(source[:500])