"""
Text Chunker - Suddivisione documenti in chunk con overlap
"""

import os
import uuid

import tiktoken


# Configurazione di default
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
DEFAULT_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def get_tokenizer(model: str = "cl100k_base"):
    """Ottiene il tokenizer di tiktoken."""
    return tiktoken.get_encoding(model)


def count_tokens(text: str, tokenizer=None) -> int:
    """Conta il numero di token in un testo."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    tokenizer=None,
) -> list[str]:
    """
    Suddivide un testo in chunk basati sui token.

    Args:
        text: Il testo da suddividere
        chunk_size: Numero massimo di token per chunk
        overlap: Numero di token di sovrapposizione tra chunk consecutivi
        tokenizer: Tokenizer tiktoken (opzionale)

    Returns:
        Lista di stringhe, ognuna rappresentante un chunk
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    tokens = tokenizer.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        start = end - overlap

        if start >= len(tokens):
            break

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Processa una lista di documenti e li suddivide in chunk.

    Args:
        documents: Lista di documenti con campi 'content', 'source', 'type'
        chunk_size: Numero massimo di token per chunk
        overlap: Numero di token di sovrapposizione

    Returns:
        Lista di chunk con campi 'chunk_id', 'text', 'source', 'metadata'
    """
    tokenizer = get_tokenizer()
    all_chunks = []

    for doc in documents:
        content = doc["content"]
        source = doc["source"]
        doc_type = doc["type"]

        text_chunks = chunk_text(content, chunk_size, overlap, tokenizer)

        for i, chunk_text_content in enumerate(text_chunks):
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text_content,
                "source": source,
                "metadata": {
                    "type": doc_type,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                },
            }
            all_chunks.append(chunk)

    return all_chunks
