"""
Embedder - Generazione embeddings via OpenAI API
"""

import os
from typing import Optional

from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class Embedder:
    """Classe per generare embeddings usando l'API OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Inizializza l'Embedder.

        Args:
            api_key: OpenAI API key (default: da variabile ambiente)
            model: Modello di embedding da usare
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile ambiente o passa api_key.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def get_embedding(self, text: str) -> list[float]:
        """
        Genera l'embedding per un singolo testo.

        Args:
            text: Il testo da convertire in embedding

        Returns:
            Lista di float rappresentanti l'embedding (1536 dimensioni)
        """
        text = text.replace("\n", " ").strip()

        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )

        return response.data[0].embedding

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Genera embeddings per una lista di testi in batch.

        Args:
            texts: Lista di testi
            batch_size: Numero di testi per batch (max 2048)

        Returns:
            Lista di embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t.replace("\n", " ").strip() for t in batch]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_chunks(self, chunks: list[dict], batch_size: int = 100) -> list[dict]:
        """
        Aggiunge l'embedding a ogni chunk.

        Args:
            chunks: Lista di chunk con campo 'text'
            batch_size: Dimensione del batch per le API

        Returns:
            Lista di chunk con campo 'embedding' aggiunto
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.get_embeddings_batch(texts, batch_size)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks
