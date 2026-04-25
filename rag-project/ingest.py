import argparse
import os
from src.utils.config import Config
from src.utils.document_loader import load_and_chunk_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_text(text_path: str):
    config = Config()

    print(f"\n[Ingestion] Loading text: {text_path}")
    loader = TextLoader(text_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = text_path

    print(f"[Ingestion] Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME
    )
    vectorstore.persist()

    print(f"[Ingestion] Done! {len(chunks)} chunks indexed in ChromaDB at '{config.CHROMA_DIR}'")
    print("[Ingestion] You can now run: python rag_app.py\n")


def ingest_pdf(pdf_path: str):
    config = Config()

    print(f"\n[Ingestion] Loading PDF: {pdf_path}")
    chunks = load_and_chunk_pdf(
        pdf_path,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    print(f"[Ingestion] Created {len(chunks)} chunks")

    print("[Ingestion] Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("[Ingestion] Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME
    )
    vectorstore.persist()

    print(f"[Ingestion] Done! {len(chunks)} chunks indexed in ChromaDB at '{config.CHROMA_DIR}'")
    print("[Ingestion] You can now run: python rag_app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to the PDF file")
    group.add_argument("--text", help="Path to a text or markdown file")
    args = parser.parse_args()

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: File not found: {args.pdf}")
            exit(1)
        if not args.pdf.lower().endswith(".pdf"):
            print("Error: --pdf requires a PDF file. Use --text for .txt or .md files.")
            exit(1)
        ingest_pdf(args.pdf)
    elif args.text:
        if not os.path.exists(args.text):
            print(f"Error: File not found: {args.text}")
            exit(1)
        if not args.text.lower().endswith((".txt", ".md")):
            print("Error: --text requires a .txt or .md file. Use --pdf for PDF files.")
            exit(1)
        ingest_text(args.text)