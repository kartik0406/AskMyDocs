from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, HTTPException
from uuid import uuid4
import tempfile
import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieve import index_documents, hybrid_search
from generate import generate_answer

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ------------------------
# Health Check
# ------------------------

@app.get("/")
def health():
    return {"status": "ok"}

# ------------------------
# Upload API
# ------------------------

@app.post("/upload")
def upload(file: UploadFile):
    try:
        doc_id = str(uuid4())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        os.remove(tmp_path)

        if not documents:
            raise HTTPException(status_code=400, detail="Empty document")

        # 🔥 IMPROVED CHUNKING
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        logging.info(f"Chunks created: {len(chunks)}")

        index_documents(chunks, doc_id)

        return {
            "message": "Indexed successfully",
            "doc_id": doc_id,
            "chunks": len(chunks)
        }

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Ask API
# ------------------------

@app.post("/ask")
def ask(query: str, doc_id: str):
    try:
        if not query or not doc_id:
            raise HTTPException(status_code=400, detail="Missing query or doc_id")

        logging.info(f"Query: {query}")
        logging.info(f"Doc ID: {doc_id}")

        docs = hybrid_search(query, doc_id)

        if not docs:
            return {
                "answer": "No relevant information found.",
                "docs": []
            }

        try:
            answer = generate_answer(query, docs)
        except Exception as e:
            logging.error(f"LLM failed: {e}")

            return {
                "answer": "LLM quota exceeded. Showing raw context.",
                "docs": docs
            }

        return {
            "answer": answer,
            "docs": docs
        }

    except Exception as e:
        logging.error(f"Ask failed: {e}")

        return {
            "answer": "Error processing request",
            "docs": [],
            "error": str(e)
        }