"""Local RAG ingestion, retrieval, LangGraph routing, and structured output."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from typing_extensions import TypedDict

import truststore
truststore.inject_into_ssl()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from prompt import GENERAL_PROMPT, POLICY_PROMPT

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
POLICY_KEYWORDS = (
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
)

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_documents: list[str]
    retrieved_ids: list[str]
    answer: str
    sources: list[str]
    confidence: float

_embedding_function: SentenceTransformerEmbeddingFunction | None = None
_collection: Any | None = None

def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device="cpu",
            normalize_embeddings=False,
        )
    return _embedding_function

def get_collection() -> Any:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=get_embedding_function(),
        )
    return _collection

def load_documents() -> list[tuple[str, str]]:
    files = sorted(DOCS_DIR.glob("doc_*.txt"))
    if len(files) != 8:
        raise ValueError(f"Expected exactly 8 policy files, found {len(files)}.")
    return [(file.stem, file.read_text(encoding="utf-8").strip()) for file in files]

def chunk_document(doc_id: str, text: str, chunk_size: int = 700) -> list[tuple[str, str]]:
    """Small fixed-size character chunks; current short documents normally remain one chunk."""
    chunks = []
    start = 0
    number = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((f"{doc_id}_chunk_{number:02d}", chunk))
            number += 1
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks

def ingest_documents() -> int:
    collection = get_collection()
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str]] = []
    for doc_id, text in load_documents():
        for chunk_id, chunk in chunk_document(doc_id, text):
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"document_id": doc_id, "chunk_id": chunk_id})
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    if collection.count() < len(ids):
        raise RuntimeError("Not all document chunks were stored in ChromaDB.")
    return len(ids)

def is_mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"

def parse_real_json(raw: str) -> AnswerResponse:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return AnswerResponse.model_validate_json(cleaned)

def real_llm_json(prompt: str) -> AnswerResponse:
    """Optional extension. Retry the real LLM up to 2 additional times after validation failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY.")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)
    corrective = ""
    last_error: Exception | None = None
    for attempt in range(3):
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt + corrective}],
        )
        raw = completion.choices[0].message.content or ""
        try:
            return parse_real_json(raw)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            corrective = (
                "\n\nCORRECTION: The prior response failed validation. "
                "Return only valid JSON with answer as string, sources as list of strings, "
                "and confidence as a number from 0 to 1."
            )
    return AnswerResponse(
        answer=f"Real-LLM output validation failed after 3 attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )

def classify_intent(state: GraphState) -> dict[str, Any]:
    query = state["query"].lower()
    if is_mock_mode():
        intent = "policy_question" if any(word in query for word in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional real-LLM classification; deterministic routing labels are still enforced.
        prompt = (
            "Classify the query as policy_question or general_question. "
            "Return JSON with answer holding only the label, sources=[], confidence. Query: "
            + state["query"]
        )
        response = real_llm_json(prompt)
        intent = "policy_question" if response.answer.strip() == "policy_question" else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState) -> dict[str, Any]:
    collection = get_collection()
    results = collection.query(
        query_texts=[state["query"]],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    ids = results["ids"][0]
    if not documents:
        return {"answer": "No relevant policy context was retrieved.", "sources": [], "confidence": 0.0}
    if is_mock_mode():
        snippet = documents[0][:200].strip()
        response = AnswerResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=ids,
            confidence=1.0,
        )
    else:
        context = "\n\n".join(f"[{chunk_id}] {text}" for chunk_id, text in zip(ids, documents))
        response = real_llm_json(POLICY_PROMPT.format(context=context, query=state["query"]))
        # Preserve grounded retrieval IDs even if the model omitted them.
        if not response.sources:
            response.sources = ids
    return {
        "retrieved_documents": documents,
        "retrieved_ids": ids,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }

def direct_answer(state: GraphState) -> dict[str, Any]:
    if is_mock_mode():
        response = AnswerResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        response = real_llm_json(GENERAL_PROMPT.format(query=state["query"]))
        response.sources = []
    return response.model_dump()

def route_after_classification(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()

GRAPH = build_graph()

def ask_question(query: str) -> AnswerResponse:
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    ingest_documents()
    state = GRAPH.invoke({"query": query.strip()})
    return AnswerResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=state["confidence"],
    )
