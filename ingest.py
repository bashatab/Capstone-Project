from rag_graph import get_collection, ingest_documents

if __name__ == "__main__":
    count = ingest_documents()
    print(f"Stored {count} chunks. Collection count: {get_collection().count()}")
