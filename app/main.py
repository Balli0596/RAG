from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from graph.rag_graph import rag_graph
from graph.rag_graph import (
    rag_graph,
    stream_rag
)
from fastapi.responses import FileResponse
import json

from fastapi.responses import (
    FileResponse,
    StreamingResponse
)

app = FastAPI(
    title="Agentic AI RAG API",
    description="RAG chatbot based on the Agentic AI ebook",
    version="1.0.0"
)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Model
# =========================

class ChatRequest(BaseModel):
    question: str


# =========================
# Health Check
# =========================

@app.get("/")
def root():

    return FileResponse(
        "static/index.html"
    )
@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    def event_generator():

        for event in stream_rag(
            request.question
        ):

            yield (
                f"data: "
                f"{json.dumps(event)}"
                f"\n\n"
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
@app.get("/pdf")
def serve_pdf():
    return FileResponse(
        "data/Ebook-Agentic-AI.pdf",
        media_type="application/pdf"
    )
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================
# Chat API
# =========================

@app.post("/chat")
def chat(request: ChatRequest):

    result = rag_graph.invoke({

        "question": request.question,

        "context": [],

        "answer": "",

        "confidence": 0.0
    })

    return {
        "answer": result["answer"],

        "confidence": result["confidence"],

        "retrieved_context": result["context"]
    }