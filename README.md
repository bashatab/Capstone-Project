# Module 3: Zepto Policy Support Assistant

## Overview

This module builds a small RAG service over eight supplied Zepto policy documents. The required graded path is deterministic mock mode. It makes no LLM call and needs no LLM API key.

## Architecture

```text
Eight text files
    ↓ ingestion: load_documents() and chunk_document()
Local all-MiniLM-L6-v2 embeddings
    ↓ embedding: SentenceTransformerEmbeddingFunction
ChromaDB collection: zepto_policy_corpus
    ↓ query embedding and top-3 cosine retrieval
LangGraph classify_intent
    ├─ policy_question  → retrieve_and_answer → validated JSON
    └─ general_question → direct_answer       → validated JSON
    ↓
FastAPI POST /ask
```

### Ingestion

`load_documents()` loads exactly `doc_01.txt` through `doc_08.txt`. `chunk_document()` creates fixed-size chunks. `ingest_documents()` embeds and upserts every chunk into the persistent ChromaDB collection.

### Embedding

`SentenceTransformerEmbeddingFunction` uses `all-MiniLM-L6-v2` locally. The model may download on its first use, after which embedding runs locally without an embedding API key.

### Retrieval

For policy questions, `retrieve_and_answer` embeds the query and retrieves the top three chunks from ChromaDB using cosine distance. Retrieval runs in both mock and optional real-LLM modes.

### Generation

- Default `MOCK_LLM=1` or unset: keyword classification and deterministic canned answers. No LLM network call.
- Optional `MOCK_LLM=0`: Groq is called using `GROQ_API_KEY`. The structured role-context-task-format-length prompt is in `prompt.py`. Invalid real-LLM JSON is retried up to two additional times.

## Prompt requirements

`prompt.py` contains the complete role, context, task, format and length template, an explicit negative constraint and a few-shot example.

## JSON schema

Every final response is validated by Pydantic:

```json
{
  "answer": "string",
  "sources": ["chunk IDs"],
  "confidence": 1.0
}
```

## Local installation

From the `support_assistant` folder:

```bash
python -m pip install -r requirements.txt
```

If a managed Windows device reports certificate errors, `truststore` is already used by the application. Do not disable SSL verification.

## Ingest and test

```bash
python ingest.py
python smoke_test.py
```

The first embedding run may download `all-MiniLM-L6-v2`. After the model is present locally, the required mock path does not use an LLM provider.

## Run FastAPI

```bash
uvicorn main:app --reload --port 8000
```

Open the interactive API documentation in a browser at `/docs`, or call:

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What is the refund policy?\"}"
```

See `example_responses.md` for one retrieval question and one general question.

## Docker

Build from `support_assistant`:

```bash
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

Then send requests to port 7860.

## Optional real LLM

The graded submission does not require this path. To test the optional extension:

```bash
set MOCK_LLM=0
set GROQ_API_KEY=your_key_here
uvicorn main:app --port 8000
```

Never commit `.env`, API keys or secrets.

## Files

```text
support_assistant/
├── docs/doc_01.txt ... doc_08.txt
├── main.py
├── rag_graph.py
├── prompt.py
├── ingest.py
├── smoke_test.py
├── example_responses.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Acceptance checklist

- [x] Eight exact policy files
- [x] Local MiniLM embeddings and persistent ChromaDB
- [x] Structured prompt with negative constraint and few-shot example
- [x] TypedDict StateGraph with three required nodes
- [x] Conditional routing after classification
- [x] Real top-3 retrieval in both modes
- [x] Deterministic mock answers
- [x] Pydantic answer, sources and confidence schema
- [x] Optional real-LLM retry logic
- [x] FastAPI POST `/ask`
- [x] Mock-mode example calls
- [x] Locally runnable Dockerfile
- [x] Written ingestion → embedding → retrieval → generation architecture

## Academic integrity

Run and understand every component. Replace representative response examples with the exact output from your own `smoke_test.py` execution and rewrite explanatory text naturally in your own words before submission.
