# Retrieval and generation

This package provides a local semantic RAG pipeline. It chunks ingested
documents, embeds the chunks with Ollama, persists vectors in a local JSON
index, retrieves the nearest chunks by cosine similarity, and asks an Ollama
chat model to answer from those passages with citations.

## Setup

1. Install the Python dependencies from the project root:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Install and start [Ollama](https://ollama.com/), then download the default
   models:

   ```powershell
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. From the project root, ask a question:

   ```powershell
   python -m rag "How do I roll back the Payment Service in production?"
   ```

The first run ingests `data/RAG-main/documents`, embeds the chunks, and saves
the local index at `data/rag_index.json`. Later runs reuse that index. Rebuild
after changing the corpus with:

```powershell
python -m rag --rebuild "What caused the duplicate payment charge incident?"
```

Set `OLLAMA_URL`, `OLLAMA_EMBEDDING_MODEL`, and
`OLLAMA_GENERATION_MODEL`, or pass corresponding command-line options, to
change the Ollama endpoint and model names. The index records the embedding
model and refuses to load with a different one; use `--rebuild` to recreate it.
The index is local and is not a replacement for a production vector database.

## Streamlit application

Install the project dependencies and start Ollama with the models above. The
Streamlit application reuses `data/rag_index.json`; it does not rebuild or
modify the index. If the index has not been created yet, run the existing CLI
once to ingest and index the documents, then launch the UI:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m rag "How do I roll back the Payment Service in production?"
streamlit run app.py
```

The sidebar selects cosine similarity, the existing hybrid/MMR retrieval,
cosine score threshold, or BM25 keyword search. Top-K limits the passages
provided to answer generation. BM25 is built from the chunks already stored in
the local index and cached for the lifetime of the Streamlit process.

## Python API

```python
from ingestion import DirectoryLoader
from rag import OllamaClient, RAGPipeline, VectorStore

client = OllamaClient()
store = VectorStore("data/rag_index.json", client)
pipeline = RAGPipeline(store, client)

if not store.is_indexed:
    documents = DirectoryLoader("data/RAG-main/documents").load()
    pipeline.index_documents(documents)

result = pipeline.answer("How do I roll back the Payment Service?")
print(result.text)
for source in result.sources:
    print(source.document.metadata, source.score)
```

The index is rebuilt by calling `pipeline.index_documents(...)` again. Model
and connection errors are raised with actionable messages; the pipeline does
not silently answer when Ollama is unavailable.
