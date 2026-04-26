mkdir askmydocs-rag
cd askmydocs-rag

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install streamlit langchain faiss-cpu sentence-transformers pypdf
pip install elasticsearch fastapi uvicorn