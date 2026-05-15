from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import load_pdf


def pdf_to_documents(text: str, title: str, source: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 30:
            continue
        docs.append(Document(
            page_content=chunk,
            metadata={"title": title, "source": source, "chunk_idx": i},
        ))
    return docs


def load_and_chunk(path: str | Path, title: str) -> List[Document]:
    text = load_pdf(path)
    return pdf_to_documents(text, title=title, source=Path(path).name)
