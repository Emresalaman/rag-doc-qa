"""Streamlit interface: pick a document, ask a question, read a cited answer."""

import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import chromadb
import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

from ask import generate_answer
from retrieve import MODEL_NAME, build_index, chunk_text, extract_text, retrieve

SAMPLE_PATH = Path(__file__).parent / "samples" / "remote_work_policy.txt"
TOP_K = 3

# The public demo spends the owner's API quota, so usage and document size are capped.
MAX_QUESTIONS_PER_SESSION = 10
MAX_QUESTIONS_PER_DAY = 100
MAX_QUESTION_LENGTH = 500
MAX_CHUNKS = 1000


class DocumentTooLargeError(Exception):
    """Raised when a document would take too long to embed on the demo's CPU."""


@st.cache_resource
def daily_usage() -> dict:
    """Return the question counter shared by all sessions of this server process."""
    return {"date": None, "count": 0, "lock": Lock()}


def reserve_question() -> str | None:
    """Count a question against the limits; return a message if a limit is reached."""
    used = st.session_state.get("questions_asked", 0)
    if used >= MAX_QUESTIONS_PER_SESSION:
        return f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per session."
    usage = daily_usage()
    today = datetime.now(timezone.utc).date()
    with usage["lock"]:
        if usage["date"] != today:
            usage.update(date=today, count=0)
        if usage["count"] >= MAX_QUESTIONS_PER_DAY:
            return "The demo reached its daily question limit. Please try again tomorrow."
        usage["count"] += 1
    st.session_state["questions_asked"] = used + 1
    return None


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
    if len(chunks) > MAX_CHUNKS:
        raise DocumentTooLargeError
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
    except DocumentTooLargeError:
        st.error("This document is too large for the demo. Try a shorter one.")
        st.stop()
    if collection is None:
        st.error("No text was found in this file. Scanned PDFs are not supported.")
        st.stop()

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="How much is the home office allowance?",
            max_chars=MAX_QUESTION_LENGTH,
        )
        submitted = st.form_submit_button("Ask")
    if not submitted or not question.strip():
        return

    limit_message = reserve_question()
    if limit_message:
        st.warning(limit_message)
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
