"""Answer questions about a document: retrieve relevant chunks, then ask Gemini."""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

from retrieve import MODEL_NAME, build_index, chunk_text, load_document, retrieve

LLM_MODEL = "gemini-3.8-flash"
# Free-tier quotas are per model, so each fallback also adds daily capacity.
FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash-lite")

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


class LLMUnavailableError(Exception):
    """Raised when every model failed; holds one HTTP status per model."""

    def __init__(self, codes: list[str]) -> None:
        super().__init__(f"All models failed with codes: {', '.join(codes)}")
        self.codes = codes


def format_context(chunks: list[str]) -> str:
    """Number the chunks so the model can cite them."""
    return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))


def error_code(error: Exception) -> str:
    """Return the HTTP status in an API error, without any other details of the message."""
    match = re.search(r"\b[45]\d{2}\b", str(error))
    return match.group() if match else "unknown"


def generate_answer(question: str, sources: list[str], api_key: str) -> tuple[str, str | None]:
    """Answer from the given chunks; return the answer and the model that wrote it."""
    # Few retries so a busy primary model fails fast and the fallback takes over.
    settings = {"google_api_key": api_key, "max_retries": 1, "timeout": 30}
    # Temperature 0 favors consistent, factual answers over creative ones.
    # The fallback models use fixed sampling defaults, so they take no temperature.
    models = [
        ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, **settings),
        *(ChatGoogleGenerativeAI(model=name, **settings) for name in FALLBACK_MODELS),
    ]
    inputs = {"context": format_context(sources), "question": question}

    # A manual loop instead of with_fallbacks, which only reports the first model's error.
    codes = []
    for model in models:
        try:
            response = (PROMPT | model).invoke(inputs)
            return response.text, response.response_metadata.get("model_name")
        except Exception as error:
            code = error_code(error)
            print(f"{model.model} failed: {type(error).__name__} {code}")
            codes.append(code)
    raise LLMUnavailableError(codes)


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

    try:
        answer, model_name = generate_answer(args.question, sources, api_key)
    except LLMUnavailableError as error:
        sys.exit(str(error))

    print(f"{answer}\n\nModel: {model_name}\nSources:")
    for i, (chunk, score) in enumerate(matches, start=1):
        print(f"[{i}] similarity={score:.3f} {chunk[:80]}...")


if __name__ == "__main__":
    main()
