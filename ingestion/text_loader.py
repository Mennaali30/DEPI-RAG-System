from datetime import datetime

from .Document import Document
from .base import DocumentLoader


class TextLoader(DocumentLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return Document(
            content=content,
            metadata={
                "source": str(self.file_path),
                "file_type": "txt",
                "created_at": datetime.now().date(),
            },
        )