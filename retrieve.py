"""Minimal retrieval pipeline: load -> chunk -> embed -> search."""

import argparse
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Multilingual model, so Turkish questions can match English text and vice versa.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# This model reads at most 128 tokens per input, so chunks stay small enough to fit.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def load_document(path: Path) -> str:
    """Return the plain text of a PDF or text file."""
    if path.suffix.lower() == ".pdf":
        pages = PdfReader(path).pages
        return "\n\n".join(page.extract_text() or "" for page in pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph and sentence breaks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def build_index(chunks: list[str], model: SentenceTransformer) -> chromadb.Collection:
    """Embed the chunks and store them in an in-memory Chroma collection."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="documents",
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=model.encode(chunks).tolist(),
    )
    return collection


def retrieve(
    question: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
) -> list[tuple[str, float]]:
    """Return the top_k chunks closest to the question with their similarity scores."""
    results = collection.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=top_k,
    )
    # Chroma returns cosine distance; similarity is 1 - distance.
    return [
        (chunk, 1 - distance)
        for chunk, distance in zip(results["documents"][0], results["distances"][0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find document chunks that match a question.")
    parser.add_argument("document", type=Path, help="Path to a PDF or text file")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=2, help="Number of chunks to return")
    args = parser.parse_args()

    chunks = chunk_text(load_document(args.document))
    model = SentenceTransformer(MODEL_NAME)
    collection = build_index(chunks, model)
    print(f"Indexed {len(chunks)} chunks from {args.document}\n")

    matches = retrieve(args.question, collection, model, min(args.top_k, len(chunks)))
    for rank, (chunk, score) in enumerate(matches, start=1):
        print(f"[{rank}] similarity={score:.3f}\n{chunk}\n")


if __name__ == "__main__":
    main()
