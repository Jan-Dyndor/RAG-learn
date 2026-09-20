import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")


test_doc = Document(
    page_content="This is a test document to verify Supabase connection.",
    metadata={"test": True},
)


def connect_to_supabase():
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="test docs",
        connection=SUPABASE_URL,
        use_jsonb=True,
    )

    return vectorstore


def verify_connection(vectorstore):
    """Verify the connection works"""

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"✅ Added test document: {ids[0]}")

        # Search for it
        results = vectorstore.similarity_search("test document", k=1)
        if results:
            print(f"✅ Search works: {results[0].page_content[:50]}...")

        # Clean up
        vectorstore.delete(ids)
        print("✅ Cleanup complete")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


vectorstore = connect_to_supabase()

verify_connection(vectorstore)
