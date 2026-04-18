from langchain.document_loaders import PyPDFLoader

def load_pdf():
    loader = PyPDFLoader("knowledge/strategies.pdf")
    documents = loader.load()
    return documents
