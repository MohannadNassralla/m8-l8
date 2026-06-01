"""Driver: bring up the Post schema and ingest the corpus.

Usage (after Weaviate is running on localhost:8080):
    python ingest.py
"""

import os

import weaviate
from sentence_transformers import SentenceTransformer

from retrieval import create_schema, index_corpus

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus.jsonl")


def main() -> None:
    client = weaviate.Client(WEAVIATE_URL)
    if not client.is_ready():
        raise SystemExit(f"Weaviate not reachable at {WEAVIATE_URL}")

    print(f"Loading embedder all-MiniLM-L6-v2 (first call downloads ~80 MB)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating schema...")
    create_schema(client)

    print(f"Ingesting corpus from {CORPUS_PATH}...")
    count = index_corpus(client, CORPUS_PATH, embedder)
    print(f"Ingested {count} objects into class 'Post'.")


if __name__ == "__main__":
    main()
