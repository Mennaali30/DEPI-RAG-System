import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        generation_model: str = "llama3.2",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post_json(
            "/api/embed",
            {"model": self.embedding_model, "input": texts},
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError(
                "Ollama returned an unexpected embedding response. "
                "Check that the selected embedding model is available."
            )

        vectors = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                raise OllamaError("Ollama returned an empty or invalid embedding.")
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as error:
                raise OllamaError("Ollama returned a non-numeric embedding.") from error
            vectors.append(vector)
        return vectors

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._post_json(
            "/api/chat",
            {
                "model": self.generation_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise OllamaError(
                "Ollama returned an unexpected chat response. "
                "Check that the selected generation model is available."
            )
        answer = message["content"].strip()
        if not answer:
            raise OllamaError("Ollama returned an empty answer.")
        return answer

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            message = details or error.reason
            raise OllamaError(
                f"Ollama request to {path} failed with HTTP {error.code}: {message}"
            ) from error
        except URLError as error:
            raise OllamaError(
                f"Could not connect to Ollama at {self.base_url}: {error.reason}. "
                "Start Ollama and verify the configured URL."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaError(
                f"Ollama returned an invalid JSON response for {path}."
            ) from error

        if not isinstance(result, dict):
            raise OllamaError(f"Ollama returned an invalid response for {path}.")
        if isinstance(result.get("error"), str):
            raise OllamaError(f"Ollama error for {path}: {result['error']}")
        return result
