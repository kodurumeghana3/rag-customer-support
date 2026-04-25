from pathlib import Path

def load_and_chunk_pdf(file_path: str, chunk_size=512, chunk_overlap=50):
    path = Path(file_path)
    suffix = path.suffix.lower()

    # ✅ Route by file type FIRST
    if suffix == ".pdf":
        docs = _load_pdf(file_path)
    elif suffix in (".txt", ".md"):
        docs = _load_text(file_path)   # ← goes here for .txt
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return _chunk(docs, chunk_size, chunk_overlap)


def _load_pdf(file_path):
    try:
        import fitz
        from langchain.schema import Document
        result = []
        pdf = fitz.open(file_path)
        for i, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                result.append(Document(
                    page_content=text,
                    metadata={
                        "source": Path(file_path).name,
                        "page": i + 1
                    }
                ))
        pdf.close()
        return result
    except ImportError:
        import pdfplumber
        from langchain.schema import Document
        result = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    result.append(Document(
                        page_content=text,
                        metadata={
                            "source": Path(file_path).name,
                            "page": i + 1
                        }
                    ))
        return result


def _load_text(file_path):
    from langchain.schema import Document
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        return []
    return [Document(
        page_content=content,
        metadata={
            "source": Path(file_path).name,
            "page": 1
        }
    )]


def _chunk(docs, chunk_size, chunk_overlap):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)