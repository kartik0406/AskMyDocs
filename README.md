# AskMyDocs - RAG (Retrieval Augmented Generation)

A full-stack application that enables users to upload PDF documents and ask questions about them using advanced Retrieval Augmented Generation (RAG) techniques powered by LangChain, Elasticsearch, Pinecone, and Google Gemini API.

## 🎯 Features

- **PDF Upload & Processing**: Upload PDF documents which are automatically parsed and indexed
- **Hybrid Search**: Combines vector search (Pinecone) with traditional search (Elasticsearch)
- **Smart Reranking**: Uses advanced reranking to improve retrieval quality
- **LLM Integration**: Powered by Google Gemini API for intelligent answer generation
- **Web Interface**: User-friendly Streamlit frontend for easy interaction
- **FastAPI Backend**: Robust REST API for all operations

## 🏗️ Project Structure

```
AskMyDocs-RAG/
├── backend/
│   ├── main.py              # FastAPI app with /upload and /ask endpoints
│   ├── config.py            # Configuration (API keys, credentials)
│   ├── ingest.py            # PDF loading and document chunking
│   ├── retrieve.py          # Hybrid search (Elasticsearch + Pinecone)
│   ├── rerank.py            # Document reranking for better relevance
│   ├── generate.py          # LLM response generation
│   └── __init__.py
├── frontend/
│   ├── app.py               # Streamlit web interface
│   └── __init__.py
├── requirements.txt         # Python dependencies
├── instructions.md          # Setup instructions
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- API Keys for:
  - Elasticsearch Cloud
  - Pinecone
  - Google Gemini API

### Installation

1. **Clone and setup virtual environment:**
   ```bash
   cd AskMyDocs-RAG
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure credentials:**
   Edit `backend/config.py` and add your:
   - Elasticsearch Cloud ID and API Key
   - Pinecone API Key and Index name
   - Google Gemini API Key

### Running the Application

**Terminal 1 - Start Backend (FastAPI):**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 10000
```

**Terminal 2 - Start Frontend (Streamlit):**
```bash
cd frontend
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

## 📚 API Endpoints

### POST `/upload`
Uploads and indexes a PDF file.
- **Body**: `multipart/form-data` with file field
- **Returns**: Success message with document ID

### POST `/ask`
Ask a question about the indexed documents.
- **Query Parameter**: `query` (string)
- **Returns**: Generated answer and source documents

## 🔄 How It Works

1. **Document Ingestion** (`ingest.py`):
   - Loads PDF files using PyPDFLoader
   - Splits documents into chunks (500 tokens, 50 overlap)
   - Prepares for indexing

2. **Indexing & Retrieval** (`retrieve.py`):
   - Generates embeddings using Sentence Transformers
   - Stores in both Pinecone (vector) and Elasticsearch (keyword search)
   - Performs hybrid search combining both results

3. **Reranking** (`rerank.py`):
   - Reorders retrieved documents by relevance
   - Selects top 5 documents for context

4. **Answer Generation** (`generate.py`):
   - Constructs prompt with question and context
   - Calls Google Gemini API for intelligent response
   - Returns generated answer with source references

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `streamlit` | Web UI framework |
| `langchain` | RAG orchestration |
| `langchain-community` | Community integrations |
| `langchain-text-splitters` | Document chunking |
| `pypdf` | PDF parsing |
| `sentence-transformers` | Embedding generation |
| `elasticsearch` | Keyword search backend |
| `pinecone` | Vector database |
| `openai` | LLM support |

## ⚙️ Configuration

Edit `backend/config.py` to configure:
- Elasticsearch Cloud credentials
- Pinecone API details
- Gemini API key

## 🐛 Troubleshooting

**Import Error: "No module named 'langchain.document_loaders'"**
- The LangChain API structure changed in v0.1.0+
- Ensure `langchain-community` and `langchain-text-splitters` are installed:
  ```bash
  pip install langchain-community langchain-text-splitters
  ```

**Cannot connect to backend**
- Ensure FastAPI server is running on port 10000
- Check firewall settings if running on different machine

**PDF upload fails**
- Verify PDF is not corrupted
- Check file size limits

## 📝 Notes

- Documents are chunked with 500-token size and 50-token overlap
- Top 5 most relevant documents are used for answer generation
- Hybrid search combines vector similarity and keyword matching
- All API keys should be stored securely (not in version control)

## 🔐 Security

⚠️ **Important**: The current `config.py` contains placeholder credentials. For production:
- Use environment variables instead of hardcoded keys
- Add `.gitignore` entry for `config.py`
- Use secrets management tools
- Never commit API keys to version control

## 📄 License

[Add your license information here]

## 👤 Author

[Add author information here]
