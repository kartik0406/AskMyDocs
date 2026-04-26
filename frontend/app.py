import streamlit as st
import requests
import os

# Determine API URL based on environment
if os.getenv("ENVIRONMENT") == "production":
    API_URL = "https://askmydocs-ad79.onrender.com"
else:
    API_URL = "http://localhost:10000"

st.set_page_config(page_title="AskMyDocs", layout="wide")

st.title("📄 AskMyDocs (Chat RAG)")

# ------------------------
# Session State
# ------------------------

if "docs" not in st.session_state:
    st.session_state.docs = {}          # {filename: doc_id}

if "messages" not in st.session_state:
    st.session_state.messages = {}      # {filename: [msg, ...]} — per-doc history

if "uploading_file" not in st.session_state:
    st.session_state.uploading_file = None  # track which file is being uploaded

# ------------------------
# Upload Section
# ------------------------

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    filename = uploaded_file.name
    # Only upload if this specific file hasn't been indexed yet
    if filename not in st.session_state.docs:
        with st.spinner(f"Uploading and indexing **{filename}**..."):
            files = {
                "file": (
                    filename,
                    uploaded_file,
                    "application/pdf"
                )
            }

            res = requests.post(f"{API_URL}/upload", files=files)

            if res.status_code == 200:
                data = res.json()
                doc_id = data["doc_id"]

                st.session_state.docs[filename] = doc_id
                st.session_state.messages[filename] = []

                st.success(f"✅ Indexed **{filename}** ({data['chunks']} chunks)")
            else:
                st.error(res.text)
    else:
        st.info(f"**{filename}** is already indexed. Select it below.")

# ------------------------
# Select Document
# ------------------------

if st.session_state.docs:
    selected_doc = st.selectbox(
        "🗂️ Select Document to Chat With",
        list(st.session_state.docs.keys())
    )

    doc_id = st.session_state.docs[selected_doc]

    # Ensure message history exists for this doc
    if selected_doc not in st.session_state.messages:
        st.session_state.messages[selected_doc] = []

    # ------------------------
    # Chat UI — per-document history
    # ------------------------

    st.divider()

    # Display chat history for the selected document only
    for msg in st.session_state.messages[selected_doc]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box (ChatGPT style)
    if query := st.chat_input(f"Ask something about {selected_doc}..."):

        # Show user message
        st.session_state.messages[selected_doc].append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                res = requests.post(
                    f"{API_URL}/ask",
                    params={
                        "query": query,
                        "doc_id": doc_id
                    }
                )

                if res.status_code == 200:
                    data = res.json()

                    answer = data.get("answer", "No answer")
                    docs = data.get("docs", [])

                    st.markdown(answer)

                    # Show sources expandable with numbered citations
                    if docs:
                        with st.expander("📚 Sources"):
                            for i, d in enumerate(docs):
                                st.markdown(f"**[Source {i+1}]**")
                                st.caption(d[:500])
                                st.divider()

                    # Save assistant response to this doc's history
                    st.session_state.messages[selected_doc].append({
                        "role": "assistant",
                        "content": answer
                    })

                else:
                    st.error(res.text)

# ------------------------
# Reset Buttons
# ------------------------

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Clear Chat History"):
        if st.session_state.docs:
            # Clear history only for the selected doc
            selected = list(st.session_state.docs.keys())
            if selected:
                for doc in selected:
                    st.session_state.messages[doc] = []
        st.rerun()

with col2:
    if st.button("🗑️ Remove All Documents"):
        st.session_state.docs = {}
        st.session_state.messages = {}
        st.rerun()