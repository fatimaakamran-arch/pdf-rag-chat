import streamlit as st
import tempfile
import os

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="RAG Chat with your PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF (RAG + Groq)")

# -----------------------------
# Session state init
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# -----------------------------
# Sidebar: API key + PDF upload
# -----------------------------
with st.sidebar:
    st.header("Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key from https://console.groq.com/keys",
    )

    model_name = st.selectbox(
        "Groq Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct"],
        index=0,
    )

    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)

    process_btn = st.button("Process PDF", type="primary", use_container_width=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# -----------------------------
# Cached embedding model loader
# (loaded once per session, reused across reruns)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def build_vectorstore(text: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    if not chunks:
        return None, 0

    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, len(chunks)


def get_groq_answer(api_key: str, model: str, question: str, context: str) -> str:
    client = Groq(api_key=api_key)

    system_prompt = (
        "You are a helpful assistant answering questions about a PDF document. "
        "Use ONLY the provided context to answer. If the answer is not in the "
        "context, say you don't know based on the document."
    )

    user_prompt = f"Context from the document:\n{context}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# -----------------------------
# Process PDF button
# -----------------------------
if process_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    else:
        with st.spinner("Extracting text and building the vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            text = extract_text_from_pdf(tmp_path)
            os.unlink(tmp_path)

            if not text.strip():
                st.sidebar.error(
                    "Could not extract any text from this PDF (it may be a scanned/image-only PDF)."
                )
            else:
                vectorstore, n_chunks = build_vectorstore(text, chunk_size, chunk_overlap)
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                st.sidebar.success(f"Indexed {n_chunks} chunks from '{uploaded_file.name}'.")

# -----------------------------
# Main chat area
# -----------------------------
if st.session_state.vectorstore is None:
    st.info("Upload a PDF and click **Process PDF** in the sidebar to get started.")
else:
    st.caption(f"Currently chatting with: **{st.session_state.pdf_name}**")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs = st.session_state.vectorstore.similarity_search(question, k=4)
                    context = "\n\n---\n\n".join(d.page_content for d in docs)

                    try:
                        answer = get_groq_answer(groq_api_key, model_name, question, context)
                    except Exception as e:
                        answer = f"Error calling Groq API: {e}"

                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
