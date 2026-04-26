import streamlit as st
import requests

API_URL = "https://askmydocs-ad79.onrender.com/"

st.set_page_config(page_title="AskMyDocs", layout="wide")

st.title("📄 AskMyDocs (Chat RAG)")

# ------------------------
# Session State
# ------------------------

if "docs" not in st.session_state:
    st.session_state.docs = {}

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# Upload Section
# ------------------------

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.uploaded:
    with st.spinner("Uploading and indexing..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }

        res = requests.post(f"{API_URL}/upload", files=files)

        if res.status_code == 200:
            data = res.json()
            doc_id = data["doc_id"]

            st.session_state.docs[uploaded_file.name] = doc_id
            st.session_state.uploaded = True

            st.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.error(res.text)

# ------------------------
# Select Document
# ------------------------

if st.session_state.docs:
    selected_doc = st.selectbox(
        "Select Document",
        list(st.session_state.docs.keys())
    )

    doc_id = st.session_state.docs[selected_doc]

    # ------------------------
    # Chat UI
    # ------------------------

    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box (ChatGPT style)
    if query := st.chat_input("Ask something about your document..."):

        # Show user message
        st.session_state.messages.append({
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

                    # Show sources expandable
                    if docs:
                        with st.expander("📚 Sources"):
                            for d in docs:
                                st.write(d)

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:
                    st.error(res.text)

# ------------------------
# Reset Button
# ------------------------

st.divider()

if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.uploaded = False
    st.session_state.docs = {}