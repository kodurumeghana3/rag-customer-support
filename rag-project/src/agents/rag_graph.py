from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage


# ================= STATE =================
class SupportState(TypedDict):
    query: str
    retrieved_docs: List[str]
    relevant_docs: List[str]
    answer: str
    confidence: float
    escalated: bool
    escalation_reason: str
    session_id: str
    messages: Annotated[list, operator.add]


# ================= INTENT =================
def intent_node(state, config):
    query = state["query"].lower()
    keywords = config["configurable"]["escalation_keywords"]

    if any(k in query for k in keywords):
        return {**state, "escalated": True, "escalation_reason": "Sensitive query"}

    return {**state, "escalated": False}


# ================= RETRIEVER =================
def retriever_node(state, config):
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=config["configurable"]["embedding_model"])

    db = Chroma(
        persist_directory=config["configurable"]["chroma_dir"],
        embedding_function=embeddings,
        collection_name=config["configurable"]["collection_name"]
    )

    docs = db.similarity_search(state["query"], k=config["configurable"]["top_k"])

    return {
        **state,
        "retrieved_docs": [d.page_content for d in docs]
    }


# ================= GRADER =================
def grader_node(state, config):
    return {
        **state,
        "relevant_docs": state["retrieved_docs"][:3]
    }


# ================= GENERATOR =================
def generator_node(state, config):
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=config["configurable"]["llm_model"],
        temperature=0,
        api_key=config["configurable"]["groq_api_key"]
    )

    context = "\n\n".join(state["relevant_docs"])

    prompt = f"""
You are a support assistant.
Use ONLY this context:

{context}

Question: {state["query"]}
"""

    try:
        res = llm.invoke(prompt)
    except Exception as e:
        return {
            **state,
            "answer": f"[LLM error] {e}",
            "confidence": 0.0
        }

    return {
        **state,
        "answer": res.content,
        "confidence": 0.85
    }


# ================= HITL =================
def hitl_node(state, config):
    return {
        **state,
        "answer": "Escalated to human support."
    }


# ================= ROUTING =================
def route_intent(state):
    return "hitl" if state["escalated"] else "retriever"


# ================= BUILD GRAPH =================
def build_graph(config_dict):
    graph = StateGraph(SupportState)

    graph.add_node("intent", intent_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("generator", generator_node)
    graph.add_node("hitl", hitl_node)

    graph.set_entry_point("intent")

    graph.add_conditional_edges(
        "intent",
        route_intent,
        {"retriever": "retriever", "hitl": "hitl"}
    )

    graph.add_edge("retriever", "grader")
    graph.add_edge("grader", "generator")
    graph.add_edge("generator", END)
    graph.add_edge("hitl", END)

    return graph.compile()


# ================= RUN QUERY (IMPORTANT FIX) =================
def run_query(graph, query: str, session_id: str = "session_001", config_dict: dict = {}) -> dict:
    
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "relevant_docs": [],
        "answer": "",
        "confidence": 0.0,
        "escalated": False,
        "escalation_reason": "",
        "session_id": session_id,
        "messages": []
    }

    # ✅ Put ALL config_dict values inside "configurable"
    langgraph_config = {
        "configurable": {
            "session_id": session_id,
            "embedding_model": config_dict.get("embedding_model", "all-MiniLM-L6-v2"),
            "llm_model": config_dict.get("llm_model", "llama-3.1-8b-instant"),
            "top_k": config_dict.get("top_k", 5),
            "escalation_keywords": config_dict.get("escalation_keywords", []),
            "collection_name": config_dict.get("collection_name", "customer_support_kb"),
            "chroma_dir": config_dict.get("chroma_dir", "./chroma_db"),
            "groq_api_key": config_dict.get("groq_api_key", "")
        }
    }

    result = graph.invoke(initial_state, config=langgraph_config)
    return result