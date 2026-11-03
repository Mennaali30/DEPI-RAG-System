import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from ingestion import DirectoryLoader, HTMLLoader, MarkdownLoader, PDFLoader


class IngestionTests(unittest.TestCase):
    def test_html_loader_extracts_visible_text_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "example.html"
            path.write_text(
                "<html><head><title>Hidden</title></head>"
                "<body><h1>Heading</h1><p>Useful <b>text</b>.</p>"
                '<img alt="diagram"><script>not content</script></body></html>',
                encoding="utf-8",
            )

            document = HTMLLoader(path).load()

            self.assertEqual(document.content, "Heading\nUseful text.\ndiagram")
            self.assertEqual(document.metadata["file_type"], "html")

    def test_markdown_loader_reads_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "guide.md"
            path.write_text("# Heading\n\nMarkdown content.", encoding="utf-8")

            document = MarkdownLoader(path).load()

            self.assertEqual(document.content, "# Heading\n\nMarkdown content.")
            self.assertEqual(document.metadata["file_type"], "md")

    def test_pdf_loader_extracts_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "empty.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with path.open("wb") as file:
                writer.write(file)

            document = PDFLoader(path).load()

            self.assertEqual(document.content, "")
            self.assertEqual(document.metadata["page_count"], 1)
            self.assertEqual(document.metadata["file_type"], "pdf")

    def test_directory_loader_loads_supported_files_and_enriches_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            documents_path = root / "documents"
            documents_path.mkdir()
            (documents_path / "GUIDE-001.txt").write_text(
                "A guide.", encoding="utf-8"
            )
            (documents_path / "raw.html").write_text(
                "<p>HTML guide.</p>", encoding="utf-8"
            )
            (documents_path / "ignored.json").write_text("{}", encoding="utf-8")
            metadata_path = root / "metadata" / "documents.json"
            metadata_path.parent.mkdir()
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "document_id": "GUIDE-001",
                            "title": "Guide",
                            "version": "1.0",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            documents = DirectoryLoader(documents_path).load()

            self.assertEqual(len(documents), 2)
            guide = next(
                document
                for document in documents
                if document.metadata.get("document_id") == "GUIDE-001"
            )
            self.assertEqual(guide.content, "A guide.")
            self.assertEqual(guide.metadata["title"], "Guide")
            self.assertEqual(guide.metadata["version"], "1.0")
            self.assertTrue(guide.metadata["source"].endswith("GUIDE-001.txt"))

    def test_directory_loader_rejects_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing"

            with self.assertRaises(NotADirectoryError):
                DirectoryLoader(missing_path).load()


if __name__ == "__main__":
    unittest.main()
