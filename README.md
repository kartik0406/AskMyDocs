# 📄 AskMyDocs — Production RAG Application

> A domain-specific **"Ask My Docs"** system with hybrid retrieval (BM25 + vector search), cross-encoder reranking, and citation enforcement — powered by Elasticsearch, Pinecone, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange?logo=google)

---

## 🎯 Use Cases

| Use Case | Description |
|----------|-------------|
| **Resume Q&A** | Upload your resume and ask "What are my skills?", "What is my phone number?" |
| **Legal Document Analysis** | Upload contracts/agreements and extract key clauses, dates, parties |
| **Research Paper Q&A** | Upload academic papers and ask about methodology, findings, conclusions |
| **HR Policy Lookup** | Upload employee handbooks and ask about leave policies, benefits, etc. |
| **Invoice/Report Analysis** | Upload financial documents and extract totals, vendor details, dates |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AskMyDocs RAG Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌───────────────┐     ┌──────────────────────────┐   │
│  │  PDF      │────▶│ PyPDFLoader   │────▶│ RecursiveCharacterText   │   │
│  │  Upload   │     │ (parsing)     │     │ Splitter (800/200)       │   │
│  └──────────┘     └───────────────┘     └────────────┬─────────────┘   │
│                                                       │                 │
│                                          ┌────────────┴────────────┐   │
│                                          ▼                         ▼   │
│                                ┌──────────────────┐  ┌──────────────┐ │
│                                │  Elasticsearch    │  │  Pinecone    │ │
│                                │  (BM25 Index)     │  │  (Vectors)   │ │
│                                │  Fuzzy Matching   │  │  Gemini Emb. │ │
│                                └────────┬─────────┘  └──────┬───────┘ │
│                                         │                    │         │
│  ┌──────────┐                ┌──────────┴────────────────────┘         │
│  │  User     │───────────────▶  Hybrid Search (α-blend fusion)        │
│  │  Query    │                └─────────────┬──────────────────         │
│  └──────────┘                               ▼                          │
│                                ┌─────────────────────────┐             │
│                                │  Cross-Encoder Reranking │             │
│                                │  (Gemini-based scoring)  │             │
│                                └────────────┬────────────┘             │
│                                             ▼                          │
│                                ┌─────────────────────────┐             │
│                                │  Gemini 2.5 Flash        │             │
│                                │  (Citation-enforced      │             │
│                                │   answer generation)     │             │
│                                └────────────┬────────────┘             │
│                                             ▼                          │
│                                ┌─────────────────────────┐             │
│                                │  Answer with [Source N]  │             │
│                                │  citations + sources     │             │
│                                └─────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

### 🔍 Hybrid Retrieval (BM25 + Vector Search)
- **BM25 (Elasticsearch)**: Keyword-based search with fuzzy matching, handles typos and partial terms
- **Vector Search (Pinecone + Gemini Embeddings)**: Semantic search using `text-embedding-004` — understands meaning, not just keywords
- **Alpha-Blended Fusion**: Min-max normalized scores combined with configurable α weighting

### 🎯 Cross-Encoder Reranking
- Gemini-based relevance scoring (0-10) for each passage against the query
- Reorders hybrid search results by true relevance
- Zero memory overhead — API-based, no local PyTorch models

### 📑 Citation Enforcement
- Source passages numbered `[Source 1]`, `[Source 2]`, etc.
- LLM forced to cite `[Source N]` after every claim
- Expandable source viewer in the UI — click to verify any citation

### 📄 Smart Document Processing
- PDF parsing via PyPDFLoader
- RecursiveCharacterTextSplitter with 800-character chunks and 200-character overlap
- Dual indexing to both Elasticsearch and Pinecone

---

## 📂 Project Structure

```
AskMyDocs-RAG/
├── backend/
│   ├── main.py              # FastAPI app — /upload and /ask endpoints
│   ├── config.py            # Environment variable configuration
│   ├── retrieve.py          # Hybrid search, embeddings, reranking
│   ├── generate.py          # Gemini LLM — citation-enforced generation
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   └── app.py               # Streamlit chat UI
├── render.yaml              # Render deployment config
├── requirements.txt         # Root dependencies (Render uses this)
├── .env                     # API keys (not committed)
├── .env.example             # Template for required env vars
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API Keys for:
  - [Elasticsearch Cloud](https://cloud.elastic.co/) (Cloud ID + API Key)
  - [Pinecone](https://www.pinecone.io/) (API Key + Index named `askmydocs`, 384 dimensions, cosine metric)
  - [Google Gemini API](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AskMyDocs-RAG.git
cd AskMyDocs-RAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   ELASTIC_CLOUD_ID=...
#   ELASTIC_API_KEY=...
#   PINECONE_API_KEY=...
#   PINECONE_INDEX=askmydocs
#   GEMINI_API_KEY=...
```

### Run Locally

**Terminal 1 — Backend (FastAPI):**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

**Terminal 2 — Frontend (Streamlit):**
```bash
cd frontend
streamlit run app.py
```

Open **http://localhost:8501** → Upload a PDF → Start asking questions!

---

## 📚 API Endpoints

### `GET /` — Health Check
```json
{"status": "ok"}
```

### `POST /upload` — Upload & Index a PDF
- **Body**: `multipart/form-data` with `file` field
- **Response**:
```json
{
  "message": "Indexed successfully",
  "doc_id": "b8a0be16-1fa6-4020-a548-13cbbaa097f6",
  "chunks": 24
}
```

### `POST /ask` — Ask a Question
- **Query Params**: `query` (string), `doc_id` (string)
- **Response**:
```json
{
  "answer": "The employee name is **KARTIK KHANNA** [Source 1].",
  "docs": ["chunk1 text...", "chunk2 text..."]
}
```

---

## 🔄 RAG Pipeline — How It Works

| Stage | Component | What It Does |
|-------|-----------|-------------|
| **1. Ingest** | `PyPDFLoader` + `RecursiveCharacterTextSplitter` | Parses PDF → 800-char chunks with 200-char overlap |
| **2. Index** | Elasticsearch + Pinecone | Dual-indexes: BM25 text index + Gemini embedding vectors (384-dim) |
| **3. Retrieve** | `hybrid_search()` | Runs BM25 (fuzzy) + vector search in parallel, α-blends normalized scores |
| **4. Rerank** | `rerank_with_gemini()` | Gemini scores each passage 0-10 for relevance, reorders by score |
| **5. Generate** | `generate_answer()` | Sends top 8 reranked chunks to Gemini 2.5 Flash with citation-enforced prompt |
| **6. Cite** | Prompt engineering | LLM must cite `[Source N]` for every claim; UI shows expandable source list |

---

## ☁️ Deployment

### Render (Backend API)

1. Push code to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your repo — Render auto-detects `render.yaml`
4. Add environment variables in Render dashboard
5. Deploy → API available at `https://your-app.onrender.com`

### Streamlit Cloud (Frontend)

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Connect repo → set main file to `frontend/app.py`
3. Go to **Settings → Secrets** and add:
   ```toml
   API_URL = "https://your-app.onrender.com"
   ```
4. Deploy → UI available at `https://your-app.streamlit.app`

---

## 📦 Tech Stack

| Technology | Role |
|-----------|------|
| **FastAPI** | Backend REST API |
| **Streamlit** | Chat-based frontend UI |
| **Elasticsearch Cloud** | BM25 keyword search with fuzzy matching |
| **Pinecone** | Vector database for semantic search |
| **Google Gemini 2.5 Flash** | LLM for answer generation + reranking |
| **Gemini embedding-001** | Embedding model (384-dim via MRL truncation, API-based) |
| **LangChain** | PDF loading + text splitting |
| **Python 3.11** | Runtime |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `No open ports detected` (Render) | Ensure start command uses `--host 0.0.0.0 --port $PORT` |
| `Out of memory` (Render 512MB) | Don't install `sentence-transformers` — use Gemini embeddings API instead |
| PDF upload fails | Check file isn't corrupted; ensure Elasticsearch index `docs` exists |
| "No relevant information found" | Re-upload PDF (needed after embedding model changes) |
| Slow first query | Gemini API cold start — subsequent queries are faster |
| `ConnectionError` on Streamlit Cloud | `API_URL` secret not set — add it in Streamlit Cloud → Settings → Secrets |

---

## 🔐 Security

- All API keys stored in `.env` (gitignored)
- `.env.example` provided as a template
- CORS enabled for Streamlit frontend
- No credentials hardcoded in source code

---

## 👤 Author

**Kartik Khanna**

---

## 📄 License

MIT License
