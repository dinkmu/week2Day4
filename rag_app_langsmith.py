import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load OpenAI and LangSmith configuration from .env before LangChain objects are created.
load_dotenv()

st.set_page_config(page_title="NewIT Multi-PDF RAG", page_icon="📚", layout="wide")
st.title("📚 NewIT Service Pvt Ltd - Multi-PDF RAG")
st.write("Upload multiple PDFs, ask questions across all of them, and verify every answer from its sources.")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6


def uploaded_files_signature(files) -> str:
    digest = hashlib.sha256()
    for uploaded_file in sorted(files, key=lambda item: item.name.lower()):
        data = uploaded_file.getvalue()
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(str(len(data)).encode("utf-8"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def load_and_split_pdfs(files) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks: list[Document] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            safe_name = Path(uploaded_file.name).name
            temp_path = Path(temp_dir) / safe_name
            temp_path.write_bytes(uploaded_file.getvalue())

            pages = PyPDFLoader(str(temp_path)).load()
            for page in pages:
                page.metadata["source"] = safe_name
                page.metadata["file_name"] = safe_name

            chunks = splitter.split_documents(pages)
            for chunk_number, chunk in enumerate(chunks, start=1):
                chunk.metadata["chunk_number"] = chunk_number
            all_chunks.extend(chunks)

    return all_chunks


def format_context(documents: list[Document]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get("file_name", document.metadata.get("source", "Unknown"))
        page_number = int(document.metadata.get("page", 0)) + 1
        blocks.append(f"[SOURCE {index}: {file_name}, page {page_number}]\n{document.page_content}")
    return "\n\n".join(blocks)


def unique_sources(documents: list[Document]) -> list[tuple[str, int]]:
    seen = set()
    sources = []
    for document in documents:
        file_name = document.metadata.get("file_name", document.metadata.get("source", "Unknown"))
        page_number = int(document.metadata.get("page", 0)) + 1
        key = (file_name, page_number)
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources


openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
langsmith_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true" or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "newit-multi-pdf-rag"

if not openai_api_key:
    st.error("OPENAI_API_KEY was not found. Add it to the .env file and restart Streamlit.")
    st.stop()

with st.sidebar:
    st.header("Monitoring")
    if langsmith_enabled and langsmith_api_key:
        st.success("LangSmith tracing: Enabled")
        st.caption(f"Project: {langsmith_project}")
        st.caption("Token usage, latency, traces, model calls, and errors are sent to LangSmith.")
    else:
        st.warning("LangSmith tracing: Disabled")
        st.caption("Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env, then restart Streamlit.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select all PDFs together. They are indexed into one FAISS store.",
)

if uploaded_files:
    current_signature = uploaded_files_signature(uploaded_files)
    indexed_signature = st.session_state.get("indexed_signature")
    st.write("**Uploaded:** " + ", ".join(file.name for file in uploaded_files))

    should_build = st.button("Build / Rebuild Index", type="primary", use_container_width=True)

    if should_build or indexed_signature != current_signature:
        with st.spinner("Reading PDFs, splitting pages, creating embeddings, and building FAISS..."):
            try:
                chunks = load_and_split_pdfs(uploaded_files)
                if not chunks:
                    st.error("No readable text was found in the uploaded PDFs.")
                    st.stop()

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                st.session_state.store = FAISS.from_documents(chunks, embeddings)
                st.session_state.chunk_count = len(chunks)
                st.session_state.file_names = [file.name for file in uploaded_files]
                st.session_state.indexed_signature = current_signature
                st.session_state.messages = []
            except Exception as exc:
                st.exception(exc)
                st.stop()

    if "store" in st.session_state and st.session_state.get("indexed_signature") == current_signature:
        st.success(f"Indexed {st.session_state.chunk_count} chunks.")

        with st.expander("Index settings"):
            st.write(f"Chunk size: {CHUNK_SIZE} characters")
            st.write(f"Chunk overlap: {CHUNK_OVERLAP} characters")
            st.write(f"Retrieved chunks per question: {TOP_K}")
            st.write(f"LangSmith project: {langsmith_project}")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Sources: " + " | ".join(f"{f} - page {p}" for f, p in message["sources"]))

        question = st.chat_input("Ask a question about the uploaded PDFs")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching all documents..."):
                    retriever = st.session_state.store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": TOP_K, "fetch_k": 18, "lambda_mult": 0.65},
                    )
                    retrieved_docs = retriever.invoke(question)
                    context = format_context(retrieved_docs)

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a careful document question-answering assistant.
Use only the supplied context. If the evidence is insufficient, say so clearly.

Rules:
1. Give a direct answer supported by the retrieved documents.
2. Compare evidence across all retrieved documents.
3. If documents disagree, explicitly state that there is a conflict and explain each document's position separately.
4. Never invent facts or sources.
5. Cite claims inline using the exact labels in the context, for example [SOURCE 1].
6. The source list shown by the application is the audit trail."""),
                        ("human", "Question:\n{question}\n\nRetrieved context:\n{context}"),
                    ])

                    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    answer = model.invoke(
                        prompt.format_messages(question=question, context=context),
                        config={
                            "run_name": "NewIT Multi-PDF RAG Answer",
                            "tags": ["newit", "multi-pdf-rag", "streamlit"],
                            "metadata": {
                                "application": "NewIT Multi-PDF RAG",
                                "document_count": len(uploaded_files),
                                "retrieved_chunk_count": len(retrieved_docs),
                            },
                        },
                    ).content
                    sources = unique_sources(retrieved_docs)

                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + " | ".join(f"{f} - page {p}" for f, p in sources))
                if len(sources) >= 5:
                    st.info("This answer retrieved five or more source pages. Consider reducing TOP_K or tuning chunk size if this happens frequently.")

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
else:
    st.info("Upload two or more PDFs to build the shared vector index.")
