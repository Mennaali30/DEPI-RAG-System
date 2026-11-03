import json
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .Document import Document
from .base import DocumentLoader
from .text_loader import TextLoader


class MarkdownLoader(TextLoader):
    def load(self) -> Document:
        document = super().load()
        document.metadata["file_type"] = "md"
        return document


class _HTMLTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _IGNORED_TAGS = {"head", "script", "style", "template", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self, tag: str, _attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")
        elif not self._ignored_depth and tag == "img":
            alt_text = dict(_attrs).get("alt")
            if alt_text:
                self.parts.extend((" ", alt_text, " "))

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(part.split()) for part in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


class HTMLLoader(DocumentLoader):
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> Document:
        parser = _HTMLTextParser()
        parser.feed(self.file_path.read_text(encoding="utf-8"))
        parser.close()
        return Document(
            content=parser.text(),
            metadata=_file_metadata(self.file_path),
        )


class PDFLoader(DocumentLoader):
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> Document:
        reader = PdfReader(str(self.file_path))
        if reader.is_encrypted:
            raise ValueError(f"Cannot ingest encrypted PDF: {self.file_path}")

        page_text = [page.extract_text() or "" for page in reader.pages]
        return Document(
            content="\n\n".join(text for text in page_text if text.strip()),
            metadata={
                **_file_metadata(self.file_path),
                "page_count": len(reader.pages),
            },
        )


class DirectoryLoader:
    _LOADERS = {
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
        ".pdf": PDFLoader,
    }

    def __init__(
        self,
        directory: str | Path,
        metadata_file: str | Path | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.metadata_file = (
            Path(metadata_file)
            if metadata_file is not None
            else self.directory.parent / "metadata" / "documents.json"
        )

    def load(self) -> list[Document]:
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Document directory not found: {self.directory}")

        metadata_by_id = self._load_metadata()
        documents = []
        for file_path in sorted(self.directory.rglob("*")):
            if not file_path.is_file():
                continue
            loader_type = self._LOADERS.get(file_path.suffix.lower())
            if loader_type is None:
                continue

            document = loader_type(file_path).load()
            record = metadata_by_id.get(file_path.stem)
            if record is not None:
                document.metadata.update(record)
                document.metadata["source"] = str(file_path)
                document.metadata["file_type"] = file_path.suffix.lower().lstrip(".")
            documents.append(document)
        return documents

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        if not self.metadata_file.is_file():
            return {}

        records = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON list in document metadata: {self.metadata_file}"
            )
        metadata_by_id = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(
                record.get("document_id"), str
            ):
                raise ValueError(
                    f"Invalid document metadata record in {self.metadata_file}"
                )
            metadata_by_id[record["document_id"]] = record
        return metadata_by_id


def _file_metadata(file_path: Path) -> dict[str, Any]:
    return {
        "source": str(file_path),
        "file_type": file_path.suffix.lower().lstrip("."),
        "created_at": datetime.now().date(),
    }
