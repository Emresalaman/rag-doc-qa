"""Answer questions about a document: retrieve relevant chunks, then ask Gemini."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

from retrieve import MODEL_NAME, build_index, chunk_text, load_document, retrieve

LLM_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.6-flash"

# The model may only use the retrieved context, which keeps answers grounded in the document.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the context below. "
            "Cite the sources you used, like [1] or [2]. "
            "If the context does not contain the answer, say that the document "
            "does not contain it. Answer in the language of the question.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_context(chunks: list[str]) -> str:
    """Number the chunks so the model can cite them."""
    return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a document.")
    parser.add_argument("document", type=Path, help="Path to a PDF or text file")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to send to the LLM")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Add it to the .env file.")

    chunks = chunk_text(load_document(args.document))
    embedder = SentenceTransformer(MODEL_NAME)
    collection = build_index(chunks, embedder)
    matches = retrieve(args.question, collection, embedder, min(args.top_k, len(chunks)))
    sources = [chunk for chunk, _ in matches]

    # Temperature 0 favors consistent, factual answers over creative ones.
    # Few retries so a busy primary model fails fast and the fallback takes over.
    primary = ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=api_key, temperature=0, max_retries=1, timeout=30
    )
    # The fallback model uses fixed sampling defaults, so it takes no temperature.
    fallback = ChatGoogleGenerativeAI(
        model=FALLBACK_MODEL, google_api_key=api_key, max_retries=1, timeout=30
    )
    chain = PROMPT | primary.with_fallbacks([fallback])
    response = chain.invoke({"context": format_context(sources), "question": args.question})

    print(f"{response.text}\n\nModel: {response.response_metadata.get('model_name')}\nSources:")
    for i, (chunk, score) in enumerate(matches, start=1):
        print(f"[{i}] similarity={score:.3f} {chunk[:80]}...")


if __name__ == "__main__":
    main()
