import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain   
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rag.vector_store import get_retriever

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key,
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

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
    "You are Lumina Telecom's Senior Retention Specialist AI. Your goal is to provide "
    "high-impact, data-driven retention strategies to help managers reduce churn.\n\n"

    "--- CUSTOMER PROFILE ---\n"
    f"{context_profile_str}\n\n"

    "--- CORPORATE STRATEGY (Knowledge Base) ---\n"
    "{context}\n\n"

    "--- INSTRUCTIONS ---\n"
    "1. **Analyze Risk**: Use the customer's specific metrics (tenure, charges, services) to explain their risk.\n"
    "2. **Evidence-Based Strategy**: Recommend EXACT retention actions based ONLY on the provided knowledge base.\n"
    "3. **Tactic Over Generic**: Don't just say 'offer a discount'; specify which discount from the KB applies.\n"
    "4. **Professional Tone**: Be empathetic, concise, and professional. Use **bolding** for emphasis.\n"

    "5. **STRICT SCOPE CONTROL**:\n"
    "- ONLY answer questions related to telecom customer retention, churn analysis, or strategies.\n"
    "- If the user asks anything unrelated (e.g., coding, general knowledge, personal questions, etc.), DO NOT answer.\n"
    "- Instead, respond with:\n"
    "  'I'm here to assist with customer retention strategies only. Please ask a relevant question about churn or retention.'\n"

    "6. **NO HALLUCINATION RULE**:\n"
    "- If the knowledge base does not contain relevant information, say:\n"
    "  'I don’t have sufficient data in the knowledge base to provide a reliable recommendation.'\n"
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





