"""FastAPI entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag_graph import AnswerResponse, ask_question, ingest_documents

class AskRequest(BaseModel):
    query: str = Field(min_length=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    count = ingest_documents()
    print(f"ChromaDB is ready with {count} embedded chunks.")
    yield

app = FastAPI(
    title="Zepto Policy Support Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    return ask_question(request.query)
