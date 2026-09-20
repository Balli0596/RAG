from graph.rag_graph import rag_graph
from vectorstore.qdrant_store import qdrant_client


try:

    question = "What is Agentic AI?"

    result = rag_graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "confidence": 0.0
    })

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieval Score:")
    print(f"{result['confidence']:.4f}")

    print("\nRetrieved Context:")
    print("=" * 70)

    for i, item in enumerate(
        result["context"],
        start=1
    ):
        print(f"\nChunk {i}")
        print(f"Page: {item['page']}")
        print(f"Score: {item['score']:.4f}")
        print(item["text"][:500])
        print("-" * 70)

finally:

    qdrant_client.close()