from .models import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")

    chunks = []
    for document in documents:
        content = document.content.strip()
        if not content:
            continue

        start = 0
        chunk_index = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            if end < len(content):
                end = _preferred_boundary(content, start, end, chunk_size)

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(
                    Document(
                        content=chunk_content,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

            if end >= len(content):
                break
            next_start = max(start + 1, end - chunk_overlap)
            while next_start < len(content) and content[next_start].isspace():
                next_start += 1
            start = next_start

    return chunks


def _preferred_boundary(content: str, start: int, end: int, chunk_size: int) -> int:
    minimum = start + chunk_size // 2
    candidates = (
        content.rfind("\n", minimum, end),
        content.rfind(". ", minimum, end),
        content.rfind(" ", minimum, end),
    )
    boundary = max(candidates)
    if boundary < minimum:
        return end
    if content[boundary] == ".":
        return boundary + 1
    return boundary
