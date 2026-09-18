import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"


@traceable(name="RAG LEARN DEMO CHAIN BASIC")
def demo_named_runs():
    llm = ChatOllama(model="llama3:8b")

    prompt = ChatPromptTemplate.from_template("Summarize {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")
    return result


demo_named_runs()
