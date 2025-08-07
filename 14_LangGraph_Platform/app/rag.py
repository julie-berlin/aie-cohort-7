"""Retrieval-Augmented Generation (RAG) utilities and tool.

This module builds an in-memory RAG pipeline that:
- Loads PDF documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using a token-aware splitter.
- Embeds chunks with OpenAI and stores vectors in an in-memory Qdrant store.
- Exposes a LangChain Tool `retrieve_information` that retrieves relevant
  context and generates a response constrained to that context.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, List

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken; used for chunk length measurement."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: List[Document]
    response: str


def _build_rag_graph(data_dir: str) -> "CompiledGraph":
    """Construct and compile a minimal RAG graph.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into token-aware chunks.
    3) Create embeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # Load PDFs from data directory (recursive)
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    # Split documents
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        # Fallback to legacy import path if available
        from langchain.text_splitter import (  # type: ignore
            RecursiveCharacterTextSplitter,
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    # Embeddings and vector store (in-memory Qdrant)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks, embedding=embedding_model, location=":memory:"
    )
    retriever = qdrant_vectorstore.as_retriever()

    # Prompt and model
    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provide context to answer the provided user query. "
        "Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    generator_llm = ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-nano"))

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
):
    """Use Retrieval Augmented Generation to retrieve information about student loan policies"""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    # Prefer returning the response string if available
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


