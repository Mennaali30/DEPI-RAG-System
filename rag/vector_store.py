
import json
import math
import re
from pathlib import Path
from typing import Any

from ingestion import Document
from rank_bm25 import BM25Okapi

from .chunking import split_documents
from .models import SearchResult
from .ollama_client import OllamaClient


class VectorStore:
    def __init__(
        self,
        index_path: str | Path,
        client: OllamaClient,
        load_existing: bool = True,
    ) -> None:
        self.index_path = Path(index_path)
        self.client = client
        self._entries: list[dict[str, Any]] = []
        self._bm25_index: BM25Okapi | None = None
        self._bm25_tokenized_documents: list[list[str]] | None = None

        if load_existing:
            self._load()

    @property
    def is_indexed(self) -> bool:
        return bool(self._entries)

    def index_documents(
        self,
        documents: list[Document],
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        batch_size: int = 32,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        chunks = split_documents(
            documents,
            chunk_size,
            chunk_overlap,
        )

        if not chunks:
            raise ValueError(
                "No non-empty documents were provided for indexing."
            )

        entries = []

        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            vectors = self.client.embed(
                [chunk.content for chunk in batch]
            )

            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding service returned an unexpected vector count."
                )

            entries.extend(
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "embedding": vector,
                }
                for chunk, vector in zip(batch, vectors)
            )

        self._validate_dimensions(entries)
        self._entries = entries
        self._bm25_index = None
        self._bm25_tokenized_documents = None
        self._save()

        return len(entries)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidates = self._similarity_candidates(query)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return [
            _result_from_entry(candidate["entry"], candidate["score"])
            for candidate in candidates[:top_k]
        ]

    def similarity_search_with_score_threshold(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be a finite cosine threshold between -1 and 1")

        candidates = self._similarity_candidates(query)
        qualifying = [
            candidate for candidate in candidates
            if candidate["score"] >= min_score
        ]
        qualifying.sort(key=lambda item: item["score"], reverse=True)
        return [
            _result_from_entry(candidate["entry"], candidate["score"])
            for candidate in qualifying[:top_k]
        ]

    def mmr_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        return self.search(query, top_k=top_k)

    def bm25_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip() or not self._entries:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        if self._bm25_tokenized_documents is None:
            self._bm25_tokenized_documents = [
                _tokenize(entry["content"])
                for entry in self._entries
            ]
            if any(self._bm25_tokenized_documents):
                self._bm25_index = BM25Okapi(self._bm25_tokenized_documents)

        if self._bm25_index is None or self._bm25_tokenized_documents is None:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if set(query_tokens).intersection(
                    self._bm25_tokenized_documents[index]
                )
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            _result_from_entry(self._entries[index], score)
            for index, score in ranked[:top_k]
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self._entries:
            raise ValueError(
                "Vector index is empty; index documents before searching."
            )

        query_vector = self.client.embed([query])[0]
        expected_dimension = len(self._entries[0]["embedding"])

        if len(query_vector) != expected_dimension:
            raise ValueError(
                "Query embedding dimensions do not match the saved index. "
                "Rebuild the index with the configured embedding model."
            )

        # Extract meaningful keywords from the query.
        query_terms = set(_tokenize(query))

        # Calculate the document frequency of each query keyword.
        document_frequency = {
            term: 0 for term in query_terms
        }

        for entry in self._entries:
            entry_terms = set(_tokenize(entry["content"]))

            for term in query_terms & entry_terms:
                document_frequency[term] += 1

        # Give more weight to terms that appear in fewer chunks.
        term_weights = {
            term: math.log(
                (len(self._entries) + 1)
                / (document_frequency[term] + 1)
            ) + 1
            for term in query_terms
        }

        total_weight = sum(term_weights.values())
        candidates = []

        for entry in self._entries:
            embedding = entry["embedding"]

            semantic_score = _cosine_similarity(
                query_vector,
                embedding,
            )

            # Apply the optional semantic similarity threshold.
            if (
                min_score is not None
                and semantic_score < min_score
            ):
                continue

            entry_terms = set(_tokenize(entry["content"]))
            matched_terms = query_terms & entry_terms

            if total_weight > 0:
                lexical_score = (
                    sum(
                        term_weights[term]
                        for term in matched_terms
                    )
                    / total_weight
                )
            else:
                lexical_score = 0.0

            # Combine semantic and lexical relevance.
            relevance_score = (
                0.65 * semantic_score
                + 0.35 * lexical_score
            )

            candidates.append(
                {
                    "entry": entry,
                    "score": semantic_score,
                    "relevance_score": relevance_score,
                    "embedding": embedding,
                }
            )

        # Rank by relevance before selecting diverse results.
        candidates.sort(
            key=lambda item: item["relevance_score"],
            reverse=True,
        )

        if not candidates:
            return []

        # Keep a wider pool so duplicate suppression has alternatives.
        candidates = candidates[:max(top_k * 4, top_k)]

        selected = []
        remaining = candidates.copy()

        relevance_weight = 0.8

        # Similarity above this threshold is treated as a near-duplicate.
        duplicate_threshold = 0.92

        while remaining and len(selected) < top_k:
            best_candidate = None
            best_mmr_score = float("-inf")

            for candidate in remaining:
                if not selected:
                    diversity_penalty = 0.0
                    is_duplicate = False
                else:
                    similarities = [
                        _cosine_similarity(
                            candidate["embedding"],
                            chosen["embedding"],
                        )
                        for chosen in selected
                    ]

                    diversity_penalty = max(similarities)

                    # Skip candidates too similar to a selected passage.
                    is_duplicate = (
                        diversity_penalty >= duplicate_threshold
                    )

                if is_duplicate:
                    continue

                mmr_score = (
                    relevance_weight
                    * candidate["relevance_score"]
                    - (1 - relevance_weight)
                    * diversity_penalty
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate

            # Stop if all remaining candidates are near-duplicates.
            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return [
            SearchResult(
                document=Document(
                    content=item["entry"]["content"],
                    metadata=item["entry"]["metadata"],
                ),
                # Keep the displayed score as cosine similarity.
                score=item["score"],
            )
            for item in selected
        ]

    def _similarity_candidates(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if not self._entries:
            raise ValueError(
                "Vector index is empty; index documents before searching."
            )

        query_vector = self.client.embed([query])[0]
        expected_dimension = len(self._entries[0]["embedding"])
        if len(query_vector) != expected_dimension:
            raise ValueError(
                "Query embedding dimensions do not match the saved index. "
                "Rebuild the index with the configured embedding model."
            )

        return [
            {
                "entry": entry,
                "score": _cosine_similarity(
                    query_vector,
                    entry["embedding"],
                ),
            }
            for entry in self._entries
        ]

    def _load(self) -> None:
        if not self.index_path.exists():
            return

        payload = json.loads(
            self.index_path.read_text(encoding="utf-8")
        )

        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
        ):
            raise ValueError(
                f"Unsupported or invalid vector index: {self.index_path}"
            )

        if (
            payload.get("embedding_model")
            != self.client.embedding_model
        ):
            raise ValueError(
                "The saved index uses a different embedding model. "
                "Rebuild it with the configured embedding model."
            )

        entries = payload.get("entries")

        if not isinstance(entries, list):
            raise ValueError(
                f"Invalid vector index entries: {self.index_path}"
            )

        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("content"), str)
                or not isinstance(entry.get("metadata"), dict)
                or not isinstance(entry.get("embedding"), list)
            ):
                raise ValueError(
                    f"Invalid vector index entry: {self.index_path}"
                )

        self._validate_dimensions(entries)
        self._entries = entries

    def _save(self) -> None:
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": 1,
            "embedding_model": self.client.embedding_model,
            "entries": self._entries,
        }

        temporary_path = self.index_path.with_suffix(
            f"{self.index_path.suffix}.tmp"
        )

        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(self.index_path)

    @staticmethod
    def _validate_dimensions(
        entries: list[dict[str, Any]],
    ) -> None:
        dimensions = set()

        for entry in entries:
            embedding = entry.get("embedding")

            if not isinstance(embedding, list) or not embedding:
                raise ValueError(
                    "Every indexed chunk must have a non-empty embedding."
                )

            try:
                vector = [
                    float(value)
                    for value in embedding
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Embeddings must contain numeric values."
                ) from error

            if any(
                not math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    "Embeddings must contain finite numeric values."
                )

            entry["embedding"] = vector
            dimensions.add(len(vector))

        if len(dimensions) > 1:
            raise ValueError(
                "All embeddings in a vector index must have equal length."
            )


def _tokenize(text: str) -> list[str]:
    text = text.lower()

    # Normalize "roll back" and "roll-back" to "rollback".
    text = re.sub(
        r"\broll[\s-]+back\b",
        "rollback",
        text,
    )

    tokens = re.findall(r"[a-z0-9]+", text)

    stopwords = {
        "a", "an", "the", "is", "are", "was", "were",
        "be", "been", "being", "do", "does", "did",
        "i", "we", "you", "what", "how", "when",
        "where", "why", "who", "which", "should",
        "would", "could", "can", "may", "might",
        "must", "if", "in", "on", "at", "to",
        "of", "for", "from", "by", "with", "and",
        "or", "but", "it", "this", "that", "my",
        "our", "your", "their", "me", "please",
    }

    normalized = []

    for token in tokens:
        if token in stopwords:
            continue

        # Normalize common plural forms.
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
 
        elif (
            len(token) > 4
            and token.endswith("s")
            and not token.endswith(("ss", "us", "is"))
        ):
            token = token[:-1]


        normalized.append(token)

    return normalized


def _result_from_entry(
    entry: dict[str, Any],
    score: float,
) -> SearchResult:
    return SearchResult(
        document=Document(
            content=entry["content"],
            metadata=entry["metadata"],
        ),
        score=score,
    )


def _cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    left_norm = math.sqrt(
        sum(value * value for value in left)
    )
    right_norm = math.sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return (
        sum(
            a * b
            for a, b in zip(left, right)
        )
        / (left_norm * right_norm)
    )