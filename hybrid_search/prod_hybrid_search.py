from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="embeddinggemma")

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content=(
            "Product SKU-7742X is our flagship router. It supports "
            "gigabit speeds and advanced QoS features."
        ),
        metadata={"type": "product"},
    ),
    Document(
        page_content=(
            "For network connectivity issues, first check the "
            "ethernet cable and router status lights."
        ),
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content=(
            "Error code E_CONN_REFUSED indicates the server "
            "rejected the connection. Check firewall settings."
        ),
        metadata={"type": "error"},
    ),
    Document(
        page_content=(
            "The authentication process requires valid credentials. "
            "Use OAuth2 for secure API access."
        ),
        metadata={"type": "auth"},
    ),
    Document(
        page_content=(
            "Router configuration guide: Access the admin panel "
            "at 192.168.1.1 to modify settings."
        ),
        metadata={"type": "config"},
    ),
    Document(
        page_content=(
            "WCAG 2.1 compliance requires all images to have "
            "alt text and sufficient color contrast."
        ),
        metadata={"type": "compliance"},
    ),
]

# Vector Store
vector_store = Chroma.from_documents(
    documents, embedding=embeddings, collection_name="my_collection"
)


# Vector retriver
vector_retriver = vector_store.as_retriever(search_kwargs={"k": 3})

# BM25 retriver
bm25_retriver = BM25Retriever.from_documents(documents, k=3)


# Ensamble retriver
ensamble_retriver = EnsembleRetriever(
    retrievers=[vector_retriver, bm25_retriver], weights=[0.5, 0.5]
)


def test_query(query, name, retriever):
    """Test a query and show results"""
    results = retriever.invoke(query)
    print(f'\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f"  {i+1}. {preview}")
    return results


# Test queries designed to challenge vector search
test_queries = [
    "SKU-7742X specifications",  # Exact product code
    "E_CONN_REFUSED error",  # Error code
    "How do I authenticate?",  # Semantic question
    "WCAG compliance",  # Acronym
    "router configuration",  # General semantic
]


for query in test_queries:
    print("=" * 60)

    # Vector search
    test_query(query, "VECTOR", vector_retriver)

    # BM25
    test_query(query, "BM25", bm25_retriver)

    # hybrid
    test_query(query, "HYBRID", ensamble_retriver)
