# Example calls in default mock mode

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

## Policy question

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What gift card denominations are available?\"}"
```

Representative raw JSON. Run `python smoke_test.py` to record the exact local retrieval order:

```json
{
  "answer": "Based on the retrieved context: Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase.",
  "sources": ["doc_07_chunk_00", "doc_03_chunk_00", "doc_02_chunk_00"],
  "confidence": 1.0
}
```

## General question

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What is Python?\"}"
```

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
