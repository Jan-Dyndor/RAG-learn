from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Sample document with distinct topics
document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""

embeddings = OllamaEmbeddings(model="embeddinggemma")


# 1 Recursive chunking
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "], chunk_size=400, chunk_overlap=50
)

recursive_chunks = recursive_splitter.split_text(document)


recursive_vectorstore = Chroma.from_texts(
    recursive_chunks, embedding=embeddings, collection_name="recursive_chunks"
)

# 2 Semantich Chunking
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_amount=90,
    breakpoint_threshold_type="percentile",
)

semantic_chunks = semantic_splitter.split_text(document)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks, embedding=embeddings, collection_name="semantic_chunks"
)


# Test queries
test_queries = [
    "How do I authenticate with OAuth2?",
    "What happens when I hit the rate limit?",
    "How are webhooks secured?",
    "What format are errors returned in?",
]


def test_retrival(query, vector_store, name):
    result = vector_store.similarity_search(query, k=1)
    print(f"{name} - Query: {query}\n\n")
    print(f"Retrived: {result[0].page_content[:150]}")
    return result[0].page_content


test_retrival(test_queries[1], recursive_vectorstore, "RECURSIVE")
print("==" * 50)
test_retrival(test_queries[1], semantic_vectorstore, "SEMANTIC")
