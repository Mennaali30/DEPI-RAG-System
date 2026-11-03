
import argparse
import os
from pathlib import Path

from ingestion import DirectoryLoader

from .ollama_client import OllamaClient
from .pipeline import RAGPipeline
from .vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about the indexed engineering document corpus."
    )
    parser.add_argument("question", help="Question to answer from the documents")
    parser.add_argument(
        "--documents",
        default="data/RAG-main/documents",
        help="Document directory to ingest when building the index",
    )
    parser.add_argument(
        "--index",
        default="data/rag_index.json",
        help="Path for the local persistent vector index",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )
    parser.add_argument(
        "--generation-model",
        default=os.environ.get("OLLAMA_GENERATION_MODEL", "llama3.2"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved passages without generating an answer",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-ingest documents and replace the saved index",
    )

    args = parser.parse_args()

    client = OllamaClient(
        base_url=args.ollama_url,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
    )

    index_path = Path(args.index)
    vector_store = VectorStore(
        index_path,
        client,
        load_existing=not args.rebuild,
    )
    pipeline = RAGPipeline(vector_store, client)

    if args.rebuild or not vector_store.is_indexed:
        documents = DirectoryLoader(args.documents).load()
        count = pipeline.index_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Indexed {count} chunks from {len(documents)} documents.")

    if args.show_context:
        results = pipeline.retrieve(args.question, top_k=args.top_k)
        print(f"\nRetrieved {len(results)} passages:\n")

        for index, result in enumerate(results, start=1):
            metadata = result.document.metadata
            title = (
                metadata.get("title")
                or metadata.get("document_id")
                or "Source"
            )
            source = metadata.get("source", "unknown source")

            print("=" * 80)
            print(f"[{index}] {title}")
            print(f"Source: {source}")
            print(f"Similarity score: {result.score:.3f}")
            print("-" * 80)
            print(result.document.content)
            print()

        return

    answer = pipeline.answer(args.question, top_k=args.top_k)
    print(f"\n{answer.text}\n\nSources:")

    for index, result in enumerate(answer.sources, start=1):
        metadata = result.document.metadata
        title = (
            metadata.get("title")
            or metadata.get("document_id")
            or "Source"
        )
        source = metadata.get("source", "unknown source")
        print(f"[{index}] {title} — {source} (score: {result.score:.3f})")


if __name__ == "__main__":
    main()
