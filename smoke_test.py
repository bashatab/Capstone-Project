"""Deterministic mock-mode demonstrations required by the rubric."""
import json
import os
os.environ["MOCK_LLM"] = "1"
from rag_graph import GRAPH, AnswerResponse, ask_question, get_collection, ingest_documents

count = ingest_documents()
assert count >= 8
assert get_collection().count() >= 8

policy_state = GRAPH.invoke({"query": "What gift card denominations are available?"})
general_state = GRAPH.invoke({"query": "What is Python?"})

assert policy_state["intent"] == "policy_question"
assert policy_state["sources"]
assert any(source.startswith("doc_07") for source in policy_state["sources"]), policy_state["sources"]
assert policy_state["answer"].startswith("Based on the retrieved context:")
assert general_state["intent"] == "general_question"
assert general_state["sources"] == []

policy = AnswerResponse(answer=policy_state["answer"], sources=policy_state["sources"], confidence=policy_state["confidence"])
general = AnswerResponse(answer=general_state["answer"], sources=general_state["sources"], confidence=general_state["confidence"])

print("POLICY QUESTION RAW JSON")
print(json.dumps(policy.model_dump(), indent=2))
print("\nGENERAL QUESTION RAW JSON")
print(json.dumps(general.model_dump(), indent=2))
print("\nAll mock-mode checks passed.")
