from dataclasses import dataclass
from ingestion import Document


@dataclass
class SearchResult:
    document: Document
    score: float


@dataclass
class RAGAnswer:
    text: str
    sources: list[SearchResult]
