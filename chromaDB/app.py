import chromadb

chroma_client = chromadb.Client()

collections = chroma_client.get_or_create_collection(name="my_collection")

documents = [
    {"id": "doc1", "text": "Hello"},
    {"id": "doc2", "text": "How are you?"},
    {"id": "doc3", "text": "Goodbye, see you!"},
]

for doc in documents:
    collections.upsert(ids=doc["id"], documents=doc["text"])


query = "Hello world!"

results = collections.query(query_texts=query, n_results=2)
print(results)
