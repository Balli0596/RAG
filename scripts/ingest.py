from ingestion.chunker import create_chunks
from vectorstore.qdrant_store import upsert_chunks


PDF_PATH = "data/Ebook-Agentic-AI.pdf"


def main():

    print("=" * 60)
    print("AGENTIC AI RAG - INGESTION")
    print("=" * 60)

    print("\n[1] Loading and chunking PDF...")

    chunks = create_chunks(PDF_PATH)

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\n[2] Generating embeddings...")

    print("\n[3] Uploading to Qdrant...")

    upsert_chunks(chunks)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()