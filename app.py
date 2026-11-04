"""Streamlit interface: pick a document, ask a question, read a cited answer."""

import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

from ask import generate_answer
from retrieve import MODEL_NAME, build_index, chunk_text, extract_text, retrieve

SAMPLE_PATH = Path(__file__).parent / "samples" / "remote_work_policy.txt"
TOP_K = 3


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedder() -> SentenceTransformer:
    """Load the model once per server process, because loading is slow."""
    return SentenceTransformer(MODEL_NAME)


# The leading underscore tells Streamlit not to hash the model for the cache key.
@st.cache_resource(show_spinner="Indexing the document...", max_entries=5)
def index_document(
    data: bytes, filename: str, _embedder: SentenceTransformer
) -> chromadb.Collection | None:
    """Index a document once per content; return None if it has no extractable text."""
    chunks = chunk_text(extract_text(data, filename))
    return build_index(chunks, _embedder) if chunks else None


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon=":page_facing_up:")
    st.title("Document Q&A")
    st.write("Ask a question about a document. Answers cite the passages they came from.")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is not set. Add it to the .env file or the host's secrets.")
        st.stop()

    source = st.radio("Document", ["Sample document", "Upload my own"], horizontal=True)
    if source == "Sample document":
        data, filename = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
        with st.expander("Read the sample document"):
            st.text(data.decode("utf-8"))
    else:
        st.warning(
            "Do not upload private documents. Parts of your file are sent to Google's "
            "Gemini free tier, which may use submitted content to improve its products."
        )
        uploaded = st.file_uploader("PDF or UTF-8 text file", type=["pdf", "txt", "md"])
        if uploaded is None:
            st.stop()
        data, filename = uploaded.getvalue(), uploaded.name

    embedder = load_embedder()
    try:
        collection = index_document(data, filename, embedder)
    except (PdfReadError, UnicodeDecodeError):
        st.error("This file could not be read. Use a valid, unencrypted PDF or a UTF-8 text file.")
        st.stop()
    if collection is None:
        st.error("No text was found in this file. Scanned PDFs are not supported.")
        st.stop()

    with st.form("question_form"):
        question = st.text_input("Your question", placeholder="How much is the home office allowance?")
        submitted = st.form_submit_button("Ask")
    if not submitted or not question.strip():
        return

    matches = retrieve(question, collection, embedder, min(TOP_K, collection.count()))
    try:
        with st.spinner("Writing the answer..."):
            answer, model_name = generate_answer(question, [chunk for chunk, _ in matches], api_key)
    except Exception as error:
        # Keep the API's details out of the page; they may be shown to anyone using the demo.
        print(f"LLM request failed: {type(error).__name__}")
        st.error("The language model is unavailable right now. Please try again in a minute.")
        st.stop()

    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Model: {model_name}")

    st.subheader("Sources")
    for rank, (chunk, score) in enumerate(matches, start=1):
        with st.expander(f"[{rank}] similarity {score:.3f}"):
            st.write(chunk)


if __name__ == "__main__":
    main()
