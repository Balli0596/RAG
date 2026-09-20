from vectorstore.qdrant_store import search_similar_chunks


query = "What is Agentic AI?"


results = search_similar_chunks(
    query,
    top_k=5
)


print("\n")
print("=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)


for i, result in enumerate(results):

    print(f"\nResult {i + 1}")

    print(
        f"Score: {result.score:.4f}"
    )

    print(
        f"Page: {result.payload.get('page')}"
    )

    print(
        f"Section: {result.payload.get('section')}"
    )

    print("\nText:")

    print(
        result.payload.get("text", "")[:1000]
    )

    print("-" * 70)