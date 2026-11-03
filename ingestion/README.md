# Document ingestion

From the project root, load the full corpus into `Document` objects:

```python
from ingestion import DirectoryLoader

documents = DirectoryLoader("data/RAG-main/documents").load()
```

The loader recursively reads `.txt`, `.md`, `.html`, `.htm`, and `.pdf` files in
sorted path order. It attaches source, file type, and load date metadata, and
adds matching entries from `data/RAG-main/metadata/documents.json` when that
file is present. Unsupported files are skipped. File and parsing errors are
raised rather than silently ignored.

Install the PDF parser dependency with `pip install -r requirements.txt`.
Individual format loaders and the original `TextLoader` are also available from
the `ingestion` package.
