"""
AI PDF Summarizer & Q&A
A Streamlit app that lets users upload a PDF and ask questions about its content
using Retrieval-Augmented Generation (RAG) with LangChain and OpenAI.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5

# Summary-specific chunking uses larger chunks so more content fits in a
# single LLM context window, with a proportionally smaller overlap.
SUMMARY_CHUNK_SIZE = 3000
SUMMARY_CHUNK_OVERLAP = 300
# Limit how many chunks are sent to the model to stay within context limits.
MAX_SUMMARY_CHUNKS = 6

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

DOMAIN_PROMPTS = {
    "General": (
        "You are a helpful assistant. Answer the user's question based solely on "
        "the provided context from the uploaded PDF. If the answer is not contained "
        "in the context, say so clearly."
    ),
    "Medical Research": (
        "You are an expert in medical research and clinical studies. "
        "Answer the user's question based solely on the provided context from the "
        "uploaded medical research paper. Use precise medical terminology where "
        "appropriate, explain abbreviations on first use, and note any limitations "
        "or caveats mentioned in the paper. If the answer is not contained in the "
        "context, say so clearly."
    ),
    "Legal Contracts": (
        "You are an expert legal analyst specialising in contract law. "
        "Answer the user's question based solely on the provided context from the "
        "uploaded legal contract. Identify relevant clauses by name or number when "
        "possible, highlight any obligations, rights, or risks, and flag ambiguous "
        "language. If the answer is not contained in the context, say so clearly. "
        "This is not legal advice."
    ),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "upload.pdf")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        reader = PdfReader(tmp_path)
        pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def build_vector_store(text: str, api_key: str) -> FAISS:
    """Split *text* into chunks and index them with FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def answer_question(
    question: str,
    vector_store: FAISS,
    api_key: str,
    domain: str,
) -> str:
    """Retrieve relevant chunks and ask the LLM to answer *question*."""
    docs = vector_store.similarity_search(question, k=TOP_K_CHUNKS)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    system_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["General"])

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Context from the PDF:\n\n{context}\n\nQuestion: {question}"
        ),
    ]
    response = llm.invoke(messages)
    return response.content


def generate_summary(text: str, api_key: str, domain: str) -> str:
    """Generate a concise summary of the entire document."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    # Use only the first MAX_SUMMARY_CHUNKS chunks to stay within context limits
    # for very large documents.
    excerpt = "\n\n---\n\n".join(chunks[:MAX_SUMMARY_CHUNKS])

    system_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["General"])
    summary_instruction = (
        "Please provide a concise summary (3-5 paragraphs) of the following "
        "document excerpt. Highlight the key points, findings, and conclusions."
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{summary_instruction}\n\n{excerpt}"),
    ]
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="AI PDF Summarizer & Q&A",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 AI PDF Summarizer & Q&A")
    st.markdown(
        "Upload a PDF document, choose a domain, and either generate a summary "
        "or ask questions about the content using RAG (Retrieval-Augmented Generation)."
    )

    # ------------------------------------------------------------------
    # Sidebar – configuration
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key. Never stored or logged.",
        )

        domain = st.selectbox(
            "Document Domain",
            options=list(DOMAIN_PROMPTS.keys()),
            help="Tailors the AI's expertise to your document type.",
        )

        st.markdown("---")
        st.markdown(
            "**Models used**\n"
            f"- Embeddings: `{EMBEDDING_MODEL}`\n"
            f"- Chat: `{CHAT_MODEL}`"
        )
        st.markdown("---")
        st.markdown(
            "ℹ️ This app uses RAG – your PDF is split into chunks, embedded, "
            "and stored in a local FAISS index so the AI can retrieve only the "
            "relevant sections when answering your question."
        )

    # ------------------------------------------------------------------
    # Main content – PDF upload
    # ------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"], help="Maximum recommended size: 50 MB"
    )

    if not uploaded_file:
        st.info("👆 Upload a PDF to get started.")
        return

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # ------------------------------------------------------------------
    # Process the PDF (cached by file name + size)
    # ------------------------------------------------------------------
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if "file_id" not in st.session_state or st.session_state.file_id != file_id:
        with st.spinner("📖 Extracting text from PDF…"):
            raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text.strip():
            st.error(
                "Could not extract any text from this PDF. "
                "It may be a scanned image-only document."
            )
            return

        with st.spinner("🔍 Building vector index…"):
            vector_store = build_vector_store(raw_text, api_key)

        st.session_state.file_id = file_id
        st.session_state.raw_text = raw_text
        st.session_state.vector_store = vector_store
        st.session_state.chat_history = []

    raw_text = st.session_state.raw_text
    vector_store = st.session_state.vector_store

    st.success(
        f"✅ **{uploaded_file.name}** loaded — "
        f"{len(raw_text):,} characters extracted."
    )

    # ------------------------------------------------------------------
    # Tabs: Summary | Q&A
    # ------------------------------------------------------------------
    tab_summary, tab_qa = st.tabs(["📋 Summary", "💬 Q&A"])

    with tab_summary:
        st.subheader("Document Summary")
        if st.button("Generate Summary", key="btn_summary"):
            with st.spinner("✍️ Generating summary…"):
                summary = generate_summary(raw_text, api_key, domain)
            st.markdown(summary)

    with tab_qa:
        st.subheader("Ask a Question")

        # Display chat history
        for entry in st.session_state.get("chat_history", []):
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])

        question = st.chat_input("Ask a question about the document…")
        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking…"):
                    answer = answer_question(
                        question, vector_store, api_key, domain
                    )
                st.markdown(answer)

            st.session_state.chat_history.append(
                {"question": question, "answer": answer}
            )

        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
