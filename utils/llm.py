import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
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

# --- LangGraph Implementation ---

class AgentState(TypedDict):
    """The state of our agentic workflow."""
    input: str
    chat_history: List[BaseMessage]
    context_profile_str: str
    documents: List[str]
    answer: str

def retrieve_node(state: AgentState):
    """
    Node to retrieve relevant documents from ChromaDB.
    """
    retriever = get_retriever()
    docs = retriever.invoke(state["input"])
    # Convert Document objects to string content for the prompt
    doc_contents = [d.page_content for d in docs]
    return {"documents": doc_contents}

def generate_node(state: AgentState):
    """
    Node to generate the expert response using retrieved context and profile.
    """
    context = "\n\n".join(state["documents"])
    
    system_prompt = (
        "You are Lumina Telecom's Senior Retention Specialist AI. Your goal is to provide "
        "high-impact, data-driven retention strategies to help managers reduce churn.\n\n"

        "--- CUSTOMER PROFILE ---\n"
        f"{state['context_profile_str']}\n\n"

        "--- CORPORATE STRATEGY (Knowledge Base) ---\n"
        "{context}\n\n"

        "--- INSTRUCTIONS ---\n"
        "1. **Analyze Risk**: Use the customer's specific metrics (tenure, charges, services) to explain their risk.\n"
        "2. **Evidence-Based Strategy**: Recommend EXACT retention actions based ONLY on the provided knowledge base.\n"
        "3. **Tactic Over Generic**: Don't just say 'offer a discount'; specify which discount from the KB applies.\n"
        "4. **Professional Tone**: Be empathetic, concise, and professional. Use **bolding** for emphasis.\n"

        "5. **STRICT SCOPE CONTROL**:\n"
        "- ONLY answer questions related to telecom customer retention, churn analysis, or strategies.\n"
        "- If the user asks anything unrelated, respond with:\n"
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
    

    chain = prompt | llm
    response = chain.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"],
        "context": context
    })
    
    return {"answer": response.content}


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

def chat_with_agent(question, chat_history, context_profile_str):
    """
    RAG conversational endpoint using LangGraph.
    """
    formatted_history = []
    for role, content in chat_history:
        if role == "human":
            formatted_history.append(HumanMessage(content=content))
        else:
            formatted_history.append(AIMessage(content=content))

    initial_state = {
        "input": question,
        "chat_history": formatted_history,
        "context_profile_str": context_profile_str,
        "documents": [],
        "answer": ""
    }
    
    result = app.invoke(initial_state)
    return result["answer"]