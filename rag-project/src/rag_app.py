import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from agents.rag_graph import build_graph, run_query
from utils.config import Config

load_dotenv()

def print_banner():
    print("\n" + "=" * 55)
    print("   RAG Customer Support Assistant")
    print("   Built with LangGraph + ChromaDB + HITL")
    print("=" * 55)
    print("Type your question, or 'quit' to exit.\n")

def main():
    print_banner()

    app_config = Config()           # your Config object
    graph = build_graph(app_config) # pass to build_graph only

    # ✅ This is the LangGraph config — must be a plain dict
    langgraph_config = {"configurable": {"session_id": "session_001"}}

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! Have a great day.")
            break

        print("\nAssistant: ", end="", flush=True)

        # ✅ Pass the dict, not Config object
        result = run_query(graph, user_input, langgraph_config)
        print(result["answer"])

        if result.get("escalated"):
            print("\n[Note: This query has been escalated to a human agent.]")

        print()

if __name__ == "__main__":
    main()