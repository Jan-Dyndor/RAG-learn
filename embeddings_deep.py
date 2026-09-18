import numpy as np
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="embeddinggemma")


def basic_embeddings():
    text = "What is Machine Learning?"
    simple_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(simple_embedding)}")
    print(f"First 5 values: {simple_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(simple_embedding):.4f}")


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embedding = embeddings.embed_documents(text)

    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search():

    # Documents

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)


# basic_embeddings()
# batch_embeddings()
