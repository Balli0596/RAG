# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **Python, LangGraph, Qdrant, BGE embeddings, Gemini, and FastAPI**.

The chatbot answers questions strictly using the content of the **Agentic AI ebook** and provides the retrieved source pages used to generate the answer.

---

## 📌 Project Overview

This project implements a RAG pipeline for querying the Agentic AI ebook.

Instead of sending the complete PDF to the LLM, the system:

1. Extracts text from the PDF.
2. Splits the document into smaller chunks.
3. Generates vector embeddings for each chunk.
4. Stores the embeddings in Qdrant.
5. Retrieves the most relevant chunks for a user question.
6. Passes the retrieved context to Gemini.
7. Generates an answer grounded only in the retrieved ebook content.
8. Returns the answer along with retrieval scores and source pages.

The application exposes the RAG pipeline through a **FastAPI backend** and provides a simple web-based chat interface.

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   Agentic AI Ebook   │
                         │        PDF           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    PDF Extraction    │
                         │       PyMuPDF        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Chunking        │
                         │ RecursiveCharacter   │
                         │    Text Splitter     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Embeddings      │
                         │ BAAI/bge-small-en-v1 │
                         │      384 dims        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Qdrant         │
                         │    Vector Store      │
                         └──────────┬───────────┘
                                    │
                                    │
                    User Question  │
                         │          │
                         ▼          │
                  ┌──────────────┐  │
                  │   FastAPI    │  │
                  │     API      │  │
                  └──────┬───────┘  │
                         │          │
                         ▼          │
                  ┌──────────────┐  │
                  │  LangGraph   │  │
                  │ RAG Workflow │  │
                  └──────┬───────┘  │
                         │
                         ▼
                  ┌──────────────┐
                  │  Retriever   │
                  │    Qdrant    │
                  └──────┬───────┘
                         │
                  Top-K Relevant
                     Chunks
                         │
                         ▼
                  ┌──────────────┐
                  │    Gemini    │
                  │     LLM      │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Final Answer │
                  │ + Sources   │
                  │ + Score     │
                  └──────────────┘
