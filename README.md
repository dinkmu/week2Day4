# NewIT Service Pvt Ltd - Multi-PDF RAG App with Sources

A Streamlit Retrieval-Augmented Generation (RAG) application that lets users upload multiple PDF files, builds one FAISS vector index from all documents, and answers questions using retrieved PDF content with filename and page-number sources.

## Features

- Upload multiple PDFs in one operation.
- Load PDF pages with `PyPDFLoader`.
- Split documents with `RecursiveCharacterTextSplitter`.
- Create embeddings with OpenAI `text-embedding-3-small`.
- Store all PDF chunks in a single FAISS vector database.
- Keep the vector store in `st.session_state`.
- Automatically rebuild the index when the uploaded document set changes.
- Display the total number of indexed chunks.
- Retrieve evidence across all uploaded documents using MMR retrieval.
- Generate answers with OpenAI `gpt-4o-mini`.
- Display source filename and page number.
- Instruct the model to explicitly report conflicting information between documents.
- Keep chat history for the active Streamlit session.

## Project Structure

```text
multi-rag/
├── rag_app.py
├── README.md
└── requirements.txt
```

For testing, you can upload PDFs such as:

```text
handbook.pdf
policy.pdf
```

## Requirements

- Python 3.10 or later recommended
- OpenAI API key
- Internet access for OpenAI embeddings and chat-completion requests

## Setup

### 1. Create the project directory

```bash
mkdir multi-rag
cd multi-rag
```

Copy these files into the directory:

- `rag_app.py`
- `README.md`
- `requirements.txt`

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

macOS/Linux:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
```

Do not hard-code the API key in `rag_app.py` or commit it to source control.

### 5. Run the application

```bash
streamlit run rag_app.py
```

Streamlit normally opens the application in the default browser. If it does not, use the local URL printed in the terminal, commonly:

```text
http://localhost:8501
```

## How to Use

1. Start the Streamlit application.
2. Select multiple PDF files with **Upload PDF files**.
3. Allow the application to build the FAISS index.
4. Confirm that the UI displays a message such as:

```text
Indexed 412 chunks.
```

5. Enter a question in the chat box.
6. Review the generated answer.
7. Check the source list under the answer for the original PDF filename and page number.

## Example Conflict Test

The RAG application is designed to expose disagreements instead of silently choosing one document.

For example, test with documents containing different leave rules:

```text
handbook.pdf:
New joiners receive 6 leave days during the first six months.
After permanent confirmation, 24 leave days are added.

policy.pdf:
New joiners receive 3 leave days during the initial employment period.
```

Ask:

```text
How many leave days do new joiners receive?
```

A good response should explain that the documents disagree, state the position from each document, and show both source files/pages so the user can verify them.

## RAG Configuration

The application uses these default values in `rag_app.py`:

```python
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6
```

The retriever uses Maximum Marginal Relevance (MMR):

```python
search_type="mmr"
search_kwargs={
    "k": 6,
    "fetch_k": 18,
    "lambda_mult": 0.65,
}
```

MMR is useful in a multi-document RAG system because it can reduce near-duplicate results and improve evidence diversity across PDFs.

## Source Metadata

`PyPDFLoader` provides page metadata. Before chunks are added to FAISS, the application also stores a clean filename:

```python
page.metadata["source"] = safe_name
page.metadata["file_name"] = safe_name
```

The loader's page value is zero-based, so the UI converts it to a normal one-based PDF page number:

```python
page_number = int(document.metadata.get("page", 0)) + 1
```

Sources are displayed in this format:

```text
Sources: handbook.pdf - page 5 | policy.pdf - page 5
```

## Troubleshooting

### `OPENAI_API_KEY is not set`

Set the environment variable in the same terminal session before running Streamlit:

```bash
export OPENAI_API_KEY="your-openai-api-key"
streamlit run rag_app.py
```

### FAISS installation problem

Upgrade pip and reinstall the requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PDF produces no useful text

`PyPDFLoader` works with PDFs containing extractable text. Image-only/scanned PDFs normally require OCR before they can be used effectively by this application.

### Too many source pages

If an answer regularly cites five or more pages, test a lower retrieval count first:

```python
TOP_K = 4
```

If retrieval is still broad, try smaller chunks, for example:

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
```

Evaluate retrieval quality after changing these values because very small chunks can lose important context.

## Important Notes

- The FAISS index is stored in Streamlit session state and is not permanently saved to disk.
- Restarting the Streamlit process requires the PDFs to be indexed again.
- OpenAI API usage may incur charges according to the OpenAI account being used.
- Retrieved sources should be checked against the original PDF for high-impact decisions.

## Assignment Goal

This project demonstrates a multi-document RAG workflow using Python, Streamlit, LangChain, FAISS, and OpenAI while maintaining source traceability at the PDF filename and page level.
