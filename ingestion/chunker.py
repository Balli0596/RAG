import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/Ebook-Agentic-AI.pdf"


def load_pdf(pdf_path: str):
    """
    Load PDF page-by-page using PyMuPDF.
    Each page becomes a LangChain Document.
    """

    pdf = fitz.open(pdf_path)

    documents = []

    for page_number, page in enumerate(pdf):
        text = page.get_text("text")

        # Clean unnecessary whitespace
        text = clean_text(text)

        if not text.strip():
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_number + 1
            }
        )

        documents.append(document)

    pdf.close()

    return documents


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """

    # Replace common PDF extraction artifacts
    text = text.replace("\uFFFD", "")
    text = text.replace("\xa0", " ")

    # Normalize spaces
    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def detect_section(text: str) -> str:
    """
    Try to detect section headings from the ebook.

    Examples:
        1.1 The Terminology Maze
        2.1 The Core Pillars: From Perception to Execution
        3.1 Structural Layers in Multi-Agent Systems
    """

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Detect headings such as:
        # 1.1 The Terminology Maze
        # 2.3 Defining Characteristics of an Agent
        parts = line.split(" ", 1)

        if len(parts) == 2:

            number = parts[0]

            # Check something like 1.1 / 2.3 / 4.2
            if (
                len(number) == 3
                and number[0].isdigit()
                and number[1] == "."
                and number[2].isdigit()
            ):
                return line

    return "Unknown"


def chunk_documents(documents):
    """
    Split documents into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    # Add section metadata
    for chunk in chunks:
        chunk.metadata["section"] = detect_section(
            chunk.page_content
        )

    return chunks


def create_chunks(pdf_path: str):
    """
    Complete PDF -> chunks pipeline.
    """

    print("Loading PDF...")

    documents = load_pdf(pdf_path)

    print(f"Pages loaded: {len(documents)}")

    print("Creating chunks...")

    chunks = chunk_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    chunks = create_chunks(PDF_PATH)

    print("\n--- SAMPLE CHUNK ---\n")

    print(chunks[10].page_content)

    print("\n--- METADATA ---\n")

    print(chunks[0].metadata)