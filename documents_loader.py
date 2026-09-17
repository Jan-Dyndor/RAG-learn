
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


# def load_text_file():

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
#         temp_file.write(b"Hello, this is simple text file \nThis file is used to test")
#         temp_file_path = temp_file.name

#     try:
#         loader = TextLoader(temp_file_path)
#         document = loader.load()

#         for doc in document:
#             print("DOC: ", doc)
#             print("DOC PAGE CONTENT: ", doc.page_content)
#     finally:
#         os.remove(path=temp_file_path)


# load_text_file()


def load_pdf():
    loader_pdf = PyPDFLoader(
        file_path="/Users/jan/Projects/RAG learning - FreeCodeCamp/RAG-learn/docs/sztuczna_inteligencja.pdf"
    )
    doc = loader_pdf.load()
    print(len(doc))


load_pdf()
