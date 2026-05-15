from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Embeddings locales — sin API key, sin rate limits, ~90MB descarga única
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def build_vectorstore(docs: List[Document]) -> FAISS:
    return FAISS.from_documents(docs, embeddings)


def get_retriever(vectorstore: FAISS) -> VectorStoreRetriever:
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.3},
    )
