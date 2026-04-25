# RAG Customer Support Assistant
Built with LangGraph + ChromaDB + Groq LLM + HITL

## Project Structure
rag-project/
├── data/                          # Knowledge base files
├── src/
│   ├── agents/rag_graph.py        # LangGraph workflow
│   ├── utils/config.py            # Configuration
│   ├── utils/document_loader.py   # PDF/TXT loader
│   ├── utils/hitl.py              # HITL escalation
│   ├── ingest.py                  # Document ingestion
│   └── rag_app.py                 # CLI chat interface
├── tests/
│   └── test_rag_pipeline.py       # Unit tests
├── .env                           # Environment variables
└── requirements.txt               # Dependencies

## Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Add your Groq API key to .env:
   GROQ_API_KEY=your_key_here

3. Ingest knowledge base:
   cd src
   python ingest.py --pdf ../data/supportmanual.txt

4. Run the chatbot:
   python rag_app.py

## Sample Questions
- What is the return policy?
- How do I reset my device?
- What is the warranty period?
- I want to speak to a human agent  (triggers HITL)
- I need a refund urgently          (triggers HITL)

## Technologies Used
- LangGraph  : Graph-based workflow orchestration
- ChromaDB   : Vector database for semantic search
- Groq LLM   : Fast LLM inference (llama3)
- HITL       : Human-in-the-Loop escalation
- sentence-transformers : Text embeddings