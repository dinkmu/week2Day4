import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Multi-PDF RAG with Sources", page_icon="📚", layout="wide")
st.title("📚 Multi-PDF RAG App with Sources")
st.write(
    "Upload several PDFs, build one FAISS index, and ask questions across all documents. "
    "Answers include the source file and page number."
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6


def uploaded_files_signature(files) -> str:
    """Create a stable signature so a changed upload set triggers re-indexing."""
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
                # Keep a clean display name instead of the temporary file path.
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
        blocks.append(
            f"[SOURCE {index}: {file_name}, page {page_number}]\n{document.page_content}"
        )
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


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OPENAI_API_KEY is not set. In the same terminal, run: "
        '`export OPENAI_API_KEY="your-key"` and restart Streamlit.'
    )
    st.stop()

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select all PDFs together. They will be indexed into one FAISS vector store.",
)

if uploaded_files:
    current_signature = uploaded_files_signature(uploaded_files)
    indexed_signature = st.session_state.get("indexed_signature")

    st.write("**Uploaded:** " + ", ".join(file.name for file in uploaded_files))

    should_build = st.button(
        "Build / Rebuild Index",
        type="primary",
        use_container_width=True,
    )

    if should_build or indexed_signature != current_signature:
        with st.spinner("Reading PDFs, splitting pages, and building the FAISS index..."):
            try:
                chunks = load_and_split_pdfs(uploaded_files)
                if not chunks:
                    st.error("No readable text was found in the uploaded PDFs.")
                    st.stop()

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                store = FAISS.from_documents(chunks, embeddings)

                st.session_state.store = store
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

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption(
                        "Sources: "
                        + " | ".join(
                            f"{file_name} - page {page_number}"
                            for file_name, page_number in message["sources"]
                        )
                    )

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

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """You are a careful document question-answering assistant.
Use only the supplied context. If the context does not contain enough information, say so clearly.

Important rules:
1. Give a direct, concise answer supported by the context.
2. Compare evidence across all retrieved documents.
3. If documents disagree, explicitly say that there is a conflict. State each document's position separately and do not silently choose one.
4. Do not invent facts or sources.
5. Cite claims inline using the exact labels supplied in the context, for example [SOURCE 1].
6. Treat the source list below the answer as the audit trail.
""",
                            ),
                            (
                                "human",
                                "Question:\n{question}\n\nRetrieved context:\n{context}",
                            ),
                        ]
                    )

                    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    answer = model.invoke(
                        prompt.format_messages(question=question, context=context)
                    ).content
                    sources = unique_sources(retrieved_docs)

                st.markdown(answer)
                if sources:
                    st.caption(
                        "Sources: "
                        + " | ".join(
                            f"{file_name} - page {page_number}"
                            for file_name, page_number in sources
                        )
                    )

                if len(sources) >= 5:
                    st.info(
                        "This answer retrieved five or more source pages. If this happens often, "
                        "reduce TOP_K or tune CHUNK_SIZE and CHUNK_OVERLAP to improve precision."
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
else:
    st.info("Upload two or more PDFs to build the shared vector index.")
