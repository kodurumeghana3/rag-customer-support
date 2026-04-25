import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import Config
from src.utils.document_loader import load_and_chunk_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_text(text_path):
    cfg = Config()

    print(f"[Ingestion] Loading text: {text_path}")

    loader = TextLoader(text_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = text_path

    print(f"[Ingestion] Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL
    )

    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=cfg.CHROMA_DIR,
        collection_name=cfg.COLLECTION_NAME
    )

    db.persist()

    print("[Ingestion] DONE")


def ingest_pdf(pdf_path):
    cfg = Config()

    print(f"[Ingestion] Loading PDF: {pdf_path}")

    chunks = load_and_chunk_pdf(
        pdf_path,
        cfg.CHUNK_SIZE,
        cfg.CHUNK_OVERLAP
    )

    print(f"[Ingestion] Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL
    )

    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=cfg.CHROMA_DIR,
        collection_name=cfg.COLLECTION_NAME
    )

    db.persist()

    print("[Ingestion] DONE")


if __name__ == "__main__":
    if "--text" in sys.argv:
        idx = sys.argv.index("--text")
        text_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not text_path:
            print("Usage: python ingest.py --text path_to_text")
            exit()
        ingest_text(text_path)
    elif "--pdf" in sys.argv:
        idx = sys.argv.index("--pdf")
        pdf_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not pdf_path:
            print("Usage: python ingest.py --pdf path_to_pdf")
            exit()
        if not os.path.exists(pdf_path):
            print(f"Error: File not found: {pdf_path}")
            exit(1)
        if not pdf_path.lower().endswith(".pdf"):
            print("Error: --pdf requires a PDF file. Use --text for text files like .txt.")
            exit(1)
        ingest_pdf(pdf_path)
    else:
        print("Usage: python ingest.py --text path_to_text or --pdf path_to_pdf")
        exit()