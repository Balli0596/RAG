from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from vectorstore.qdrant_store import (
    search_similar_chunks
)


load_dotenv()


# ==========================================
# State
# ==========================================

class RAGState(TypedDict):

    question: str

    context: list[dict]

    answer: str

    confidence: float


# ==========================================
# LLM
# ==========================================

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

# ==========================================
# Retrieval Node
# ==========================================
def stream_rag(question: str):
    """
    Retrieve context and stream the LLM response.
    """

    # -------------------------
    # Retrieve context
    # -------------------------

    results = search_similar_chunks(
        question,
        top_k=5
    )

    context = []

    for result in results:

        page = result.payload.get("page", 0)

        context.append({
            "text": result.payload.get(
                "text",
                ""
            ),
            "page": page,
            "section": result.payload.get(
                "section",
                "Unknown"
            ),
            "score": float(result.score),

            "source": f"/pdf#page={page}"
        })

    confidence = (
        context[0]["score"]
        if context
        else 0.0
    )

    # -------------------------
    # Build context
    # -------------------------

    context_text = "\n\n".join(
        [
            f"""
Page: {item['page']}

{item['text']}
"""
            for item in context
        ]
    )

    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are a RAG assistant for the Agentic AI ebook.

Answer the user's question ONLY using the
provided context.

Do NOT use outside knowledge.

If the answer cannot be found in the context,
respond exactly:

"I couldn't find this information in the provided ebook."

User Question:

{question}

Retrieved Context:

{context_text}

Answer:
"""

    # -------------------------
    # Stream Gemini
    # -------------------------

    for chunk in llm.stream(prompt):

        if not chunk.content:
            continue

        content = chunk.content

        # Gemini/LangChain may return content
        # as a list of content blocks.
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):

                    text_parts.append(item)

                elif isinstance(item, dict):

                    if item.get("type") == "text":

                        text = item.get("text", "")

                        if text:
                            text_parts.append(text)

            content = "".join(text_parts)

        elif not isinstance(content, str):

            content = str(content)


        if content:

            yield {
                "type": "token",
                "content": content
            }
    # -------------------------
    # Send metadata at the end
    # -------------------------

    yield {
        "type": "metadata",
        "confidence": confidence,
        "context": context
    }
def retrieve_node(state: RAGState):

    question = state["question"]

    results = search_similar_chunks(
        question,
        top_k=5
    )

    context = []

    for result in results:

        page = result.payload.get("page", 0)

        context.append({
            "text": result.payload.get(
                "text",
                ""
            ),
            "page": page,
            "section": result.payload.get(
                "section",
                "Unknown"
            ),
            "score": float(result.score),
            "source": f"/pdf#page={page}"
        })

    confidence = (
        context[0]["score"]
        if context
        else 0.0
    )

    return {
        "context": context,
        "confidence": confidence
    }


# ==========================================
# Generation Node
# ==========================================

def generate_node(state: RAGState):

    question = state["question"]

    context = state["context"]

    context_text = "\n\n".join(

        [
            f"""
Page: {item['page']}

{item['text']}
"""
            for item in context
        ]
    )

    prompt = f"""
You are a RAG assistant for the Agentic AI ebook.

Answer the user's question ONLY using the
provided context.

Do NOT use outside knowledge.

If the answer cannot be found in the context,
respond exactly:

"I couldn't find this information in the provided ebook."

User Question:

{question}

Retrieved Context:

{context_text}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ==========================================
# Build Graph
# ==========================================

def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()


rag_graph = build_graph()