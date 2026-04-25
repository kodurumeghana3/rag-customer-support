class Config:
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "rag_collection"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_MODEL = "llama-3.1-8b-instant"

    TOP_K = 3
    CONFIDENCE_THRESHOLD = 0.6

    ESCALATION_KEYWORDS = [
        "hack", "fraud", "legal", "sue", "breach", "stolen"
    ]

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150