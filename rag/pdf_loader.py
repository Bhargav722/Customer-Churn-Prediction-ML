from langchain_community.document_loaders import PyPDFLoader

def load_pdf():
    loader = PyPDFLoader("knowledge/strategies.pdf")
    documents = loader.load()
    return documents

def get_strategy_text():
    documents = load_pdf()
    text = "\n\n".join([doc.page_content for doc in documents])
    return text
