"""
MongoDB Client - Interazione con MongoDB locale (Compass)
Ricerca vettoriale implementata in Python con similarità coseno
"""

import math
import os
from typing import Optional

from pymongo import MongoClient


DEFAULT_DB_NAME = "rag_db"
DEFAULT_COLLECTION_NAME = "chunks"


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calcola la similarità coseno tra due vettori."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


class MongoDBClient:
    """Classe per interagire con MongoDB locale."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: str = DEFAULT_DB_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """
        Inizializza la connessione a MongoDB.

        Args:
            uri: Connection string MongoDB (default: localhost:27017)
            db_name: Nome del database
            collection_name: Nome della collection
        """
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        if not self.uri:
            raise ValueError("MONGODB_URI non trovata.")

        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Test connessione
        try:
            self.client.server_info()
        except Exception as e:
            raise ConnectionError(f"Impossibile connettersi a MongoDB: {e}")

    def insert_chunks(self, chunks: list[dict]) -> int:
        """
        Inserisce chunk con embeddings nella collection.

        Args:
            chunks: Lista di chunk con campi 'chunk_id', 'text', 'embedding', 'source', 'metadata'

        Returns:
            Numero di documenti inseriti
        """
        if not chunks:
            return 0

        result = self.collection.insert_many(chunks)
        return len(result.inserted_ids)

    def vector_search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """
        Esegue una ricerca vettoriale calcolando la similarità coseno in Python.

        Args:
            query_embedding: Embedding della query (1536 dimensioni)
            k: Numero di risultati da restituire

        Returns:
            Lista di chunk più simili con score
        """
        # Recupera tutti i documenti con embedding
        all_docs = list(self.collection.find(
            {"embedding": {"$exists": True}},
            {"_id": 0, "chunk_id": 1, "text": 1, "source": 1, "metadata": 1, "embedding": 1}
        ))

        if not all_docs:
            return []

        # Calcola similarità per ogni documento
        scored_docs = []
        for doc in all_docs:
            embedding = doc.get("embedding", [])
            if embedding:
                score = cosine_similarity(query_embedding, embedding)
                scored_docs.append({
                    "chunk_id": doc.get("chunk_id"),
                    "text": doc.get("text"),
                    "source": doc.get("source"),
                    "metadata": doc.get("metadata"),
                    "score": score,
                })

        # Ordina per score decrescente e prendi i top k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:k]

    def delete_all(self) -> int:
        """
        Elimina tutti i documenti dalla collection.

        Returns:
            Numero di documenti eliminati
        """
        result = self.collection.delete_many({})
        return result.deleted_count

    def get_stats(self) -> dict:
        """
        Restituisce statistiche sulla collection.

        Returns:
            Dizionario con statistiche
        """
        total_chunks = self.collection.count_documents({})

        sources_pipeline = [
            {"$group": {"_id": "$source"}},
            {"$count": "total"},
        ]
        sources_result = list(self.collection.aggregate(sources_pipeline))
        total_sources = sources_result[0]["total"] if sources_result else 0

        return {
            "total_chunks": total_chunks,
            "total_documents": total_sources,
            "database": self.db.name,
            "collection": self.collection.name,
        }

    def close(self):
        """Chiude la connessione al database."""
        self.client.close()
