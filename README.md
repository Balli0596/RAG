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
# 💬 Sample Questions & Answers

The following examples demonstrate the chatbot's ability to retrieve
relevant information from the Agentic AI ebook and generate
document-grounded answers.

---

## 1. What is Agentic AI?

### Question

> What is Agentic AI?

### Answer

> Agentic AI refers to systems capable of autonomous decision-making and
> action in pursuit of specific objectives. These systems can perceive,
> decide, and act independently, transitioning from standalone tools to
> interconnected systems.

**Retrieval Score:** `0.8561`

---

## 2. What are multi-agent systems?

### Question

> What are multi-agent systems?

### Answer

> Multi-agent systems (MAS) are a category of agentic systems that excel
> in tasks requiring diverse feedback and parallel task execution. The key
> idea behind MAS is to orchestrate multiple autonomous atomic agents to
> accomplish tasks. These systems are increasingly utilized for their
> ability to collaborate and adapt in real-time, making them suitable for
> complex, dynamic tasks.
>
> A system qualifies as a multi-agent AI system when it contains
> foundational agents, one or more worker and utility agents, clear goals,
> and a communication mechanism for agents to interact. MAS allow
> independent agent operation, facilitating dynamic task allocation and
> parallel processing across multiple sub-goals derived from an overall
> goal. They can leverage feedback from other agents to verify their work
> and improve task outcomes.

**Retrieval Score:** `0.8498`

---

## 3. Describe AI agents and its applications

### Question

> Describe AI agents and its applications

### Answer

> Agentic AI refers to systems capable of autonomous decision-making and
> action in pursuit of specific objectives. These systems can perceive,
> decide, and act independently, transitioning from standalone tools to
> interconnected systems.
>
> Applications of AI agents include:
>
> - **Early Software Agents:** Advanced tactical decision-making,
>   focusing on isolated tasks within workflows. They revolutionized
>   distributed computing with object-oriented paradigms and laid the
>   foundation for modern microservices.
>
> - **Intelligent Agents:** Enhanced workflow automation and introduced
>   rule-based decision systems in enterprise environments.
>
> - **Autonomous Agents (Modern Agents):** Sense and act on their
>   environment to pursue long-term goals, influencing future states for
>   continuous progress. They enable self-driving systems, smart
>   assistants, and adaptive industrial automation with continuous
>   learning.

**Retrieval Score:** `0.8325`

---

## 4. Out-of-Context Question

### Question

> What is the population of India?

### Answer

> I couldn't find this information in the provided ebook.

This demonstrates the grounding behavior of the RAG pipeline. The
chatbot is designed to answer questions using the provided Agentic AI
ebook rather than relying on external knowledge.

---

## 🔎 What These Examples Demonstrate

| Test | Demonstrates |
|---|---|
| What is Agentic AI? | Basic semantic retrieval |
| What are multi-agent systems? | Retrieval of detailed conceptual information |
| Describe AI agents and its applications | Retrieval of multiple related concepts |
| Population of India | Out-of-context / grounding behavior |

The chatbot returns the generated answer along with the retrieval score
and retrieved document context, making the RAG pipeline inspectable
during development.
