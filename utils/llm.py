import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rag.vector_store import get_retriever

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Use ChatGoogleGenerativeAI wrapper
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)

def get_llm_response(prompt):
    """Fallback legacy support for simple prompts."""
    return llm.invoke(prompt).content

def chat_with_agent(question, chat_history, context_profile_str):
    """
    RAG conversational endpoint.
    Retrieves from Chroma DB and passes chat history.
    """
    retriever = get_retriever()
    
    system_prompt = (
        "You are an expert Telecom Retention Strategist AI.\n"
        "You are chatting with a user (manager/support agent) about a specific customer.\n\n"
        "--- Customer Profile ---\n"
        f"{context_profile_str}\n\n"
        "--- Company Retention Strategies (Knowledge Base) ---\n"
        "{context}\n\n"
        "--- Instructions ---\n"
        "Answer the user's question directly based on the knowledge base and the customer's profile.\n"
        "Be helpful, professional, and concise. Do NOT invent retention strategies."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    response = retrieval_chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"]
