import os
from dotenv import load_dotenv
from src.agents.rag_graph import build_graph, run_query
from src.ingest import ingest_text, ingest_pdf
from src.utils.config import Config

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env. Please add your Groq API key to the project's .env file.")


def print_banner():
    print("\n" + "=" * 55)
    print("   RAG Customer Support Assistant")
    print("   Built with LangGraph + ChromaDB + HITL")
    print("=" * 55)
    print("Type your question, 'upload <path>' to ingest files, 'help' for commands, or 'quit' to exit.\n")


def main():
    print_banner()

    config = Config()
    config_dict = {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "top_k": config.TOP_K,
        "escalation_keywords": config.ESCALATION_KEYWORDS,
        "collection_name": config.COLLECTION_NAME,
        "chroma_dir": config.CHROMA_DIR,
        "groq_api_key": os.getenv("GROQ_API_KEY")
    }
    graph = build_graph(config_dict)

    session_id = "session_001"

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nGoodbye! Have a great day.")
            break
        except KeyboardInterrupt:
            print("\nGoodbye! Have a great day.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! Have a great day.")
            break

        if user_input.lower() in ["help", "h", "?"]:
            print("\nCommands:")
            print("  help          Show this message")
            print("  upload <path> Ingest a .txt, .md, or .pdf file")
            print("  quit          Exit the assistant\n")
            continue

        if user_input.lower().startswith("upload "):
            path = user_input[7:].strip().strip('"').strip("'")
            if not path:
                print("Please provide a file path after 'upload'.")
                continue
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                ingest_pdf(path)
            elif ext in [".txt", ".md"]:
                ingest_text(path)
            else:
                print("Unsupported upload type. Use a .txt, .md, or .pdf file.")
            continue

        print("\nAssistant: ", end="", flush=True)
        result = run_query(graph, user_input, session_id)
        print(result["answer"])

        if result.get("escalated"):
            print("\n[Note: This query has been escalated to a human agent.]")

        print()


if __name__ == "__main__":
    main()