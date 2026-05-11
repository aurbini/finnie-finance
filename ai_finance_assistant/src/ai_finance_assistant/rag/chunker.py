from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_finance_assistant.core.settings import RAGConf


def chunk_documents(docs: list[Document], rag: RAGConf) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag.chunk_size,
        chunk_overlap=rag.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
