from dotenv import load_dotenv

load_dotenv()

# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# from langchain_openai import ChatOpenAI


def main():
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # response = llm.invoke("Tell me capital of France")
    # print(response)

    llm = ChatOllama(model="llama3:8b")
    response = llm.invoke("Tell me capital of France")
    print(response)


if __name__ == "__main__":
    main()
