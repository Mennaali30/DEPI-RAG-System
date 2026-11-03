import os
from pathlib import Path
from typing import Any

import streamlit as st

from rag import OllamaClient, OllamaError, RAGPipeline, SearchResult, VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "data" / "rag_index.json"
RETRIEVAL_METHODS = (
    "Similarity Search",
    "MMR",
    "Similarity Score Threshold",
    "BM25",
)


@st.cache_resource(show_spinner="Loading the local RAG index...")
def load_pipeline(
    index_path: str,
    index_modified_ns: int,
    index_size: int,
    ollama_url: str,
    embedding_model: str,
    generation_model: str,
) -> RAGPipeline:
    del index_modified_ns, index_size
    client = OllamaClient(
        base_url=ollama_url,
        embedding_model=embedding_model,
        generation_model=generation_model,
    )
    return RAGPipeline(VectorStore(index_path, client), client)


def retrieve(
    pipeline: RAGPipeline,
    question: str,
    method: str,
    top_k: int,
    threshold: float,
) -> list[SearchResult]:
    if method == "Similarity Search":
        return pipeline.vector_store.similarity_search(question, top_k=top_k)
    if method == "MMR":
        return pipeline.vector_store.mmr_search(question, top_k=top_k)
    if method == "Similarity Score Threshold":
        return pipeline.vector_store.similarity_search_with_score_threshold(
            question,
            top_k=top_k,
            min_score=threshold,
        )
    if method == "BM25":
        return pipeline.vector_store.bm25_search(question, top_k=top_k)
    raise ValueError(f"Unsupported retrieval method: {method}")


def _source_title(source: SearchResult) -> tuple[str, str]:
    metadata: dict[str, Any] = source.document.metadata
    source_path = str(metadata.get("source", ""))
    title = (
        metadata.get("title")
        or metadata.get("document_id")
        or (Path(source_path).name if source_path else "Source")
    )
    return str(title), source_path


def _show_sources(
    sources: list[SearchResult],
    score_label: str,
) -> None:
    st.markdown("#### Retrieved Sources")
    if not sources:
        st.info("No passages were retrieved for this question.")
        return

    for index, result in enumerate(sources, start=1):
        title, source_path = _source_title(result)
        with st.expander(f"[{index}] {title}"):
            if source_path:
                st.caption(f"Source: {source_path}")
            st.caption(f"{score_label}: {result.score:.4f}")
            st.text(result.document.content)


def main() -> None:
    st.set_page_config(
        page_title="RAG Knowledge Assistant",
        page_icon="📚",
        layout="wide",
    )
    st.title("RAG Knowledge Assistant")
    st.write(
        "Ask questions about the indexed engineering documents. "
        "Answers are grounded in retrieved passages and include source citations."
    )

    with st.sidebar:
        st.header("Retrieval Settings")
        method = st.selectbox("Retrieval Method", RETRIEVAL_METHODS)
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
        threshold = 0.35
        if method == "Similarity Score Threshold":
            threshold = st.slider(
                "Similarity Threshold",
                min_value=-1.0,
                max_value=1.0,
                value=0.35,
                step=0.01,
                format="%.2f",
            )
        st.caption(f"Selected: {method} · Top-K: {top_k}")

    try:
        index_stat = INDEX_PATH.stat()
    except FileNotFoundError:
        st.error(
            f"The vector index is missing: {INDEX_PATH}. "
            "Build it once with the existing CLI before opening the app."
        )
        st.code(
            '.\\.venv\\Scripts\\python.exe -m rag "How do I roll back a payment?"',
            language="powershell",
        )
        st.stop()

    try:
        pipeline = load_pipeline(
            str(INDEX_PATH),
            index_stat.st_mtime_ns,
            index_stat.st_size,
            os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            os.environ.get("OLLAMA_GENERATION_MODEL", "llama3.2"),
        )
    except (OSError, ValueError) as error:
        st.error(f"Could not load the existing vector index: {error}")
        st.exception(error)
        st.stop()

    if not pipeline.vector_store.is_indexed:
        st.error(
            "The vector index exists but contains no passages. "
            "Index the documents with the existing CLI, then reload this page."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message["role"] == "assistant" and "sources" in message:
                st.caption(
                    f"Retrieval: {message['method']} · "
                    f"Top-K: {message['top_k']}"
                )
                _show_sources(message["sources"], message["score_label"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    if not question.strip():
        st.warning("Enter a question before sending.")
        return

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    score_label = "BM25 score" if method == "BM25" else "Cosine similarity"
    try:
        sources = retrieve(pipeline, question, method, top_k, threshold)
        if not sources:
            if method == "Similarity Score Threshold":
                response = (
                    "No passages met the selected cosine similarity threshold. "
                    "Lower the threshold or try another retrieval method."
                )
            elif method == "BM25":
                response = (
                    "No keyword-matching passages were found. "
                    "Try different terms or use a semantic retrieval method."
                )
            else:
                response = "No passages were retrieved for this question."
        else:
            response = pipeline.answer_with_sources(question, sources).text

        message = {
            "role": "assistant",
            "text": response,
            "sources": sources,
            "method": method,
            "top_k": top_k,
            "score_label": score_label,
        }
        st.session_state.messages.append(message)
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Retrieval: {method} · Top-K: {top_k}")
            _show_sources(sources, score_label)
    except (OllamaError, ValueError) as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": f"Unable to complete the request: {error}",
            }
        )
        with st.chat_message("assistant"):
            st.error(f"Unable to complete the request: {error}")
            st.exception(error)


if __name__ == "__main__":
    main()
