"""
RAG Pipeline - Orchestrazione completa del sistema RAG
"""

import os
from typing import Optional

from openai import OpenAI

from .loaders import load_directory
from .chunker import chunk_documents
from .embedder import Embedder
from .mongodb_client import MongoDBClient


DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))


class RAGPipeline:
    """Pipeline completa per il sistema RAG."""

    def __init__(
        self,
        mongodb_client: Optional[MongoDBClient] = None,
        embedder: Optional[Embedder] = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Inizializza la pipeline RAG.

        Args:
            mongodb_client: Client MongoDB (default: crea nuovo)
            embedder: Embedder per i vettori (default: crea nuovo)
            chat_model: Modello per le risposte
            temperature: Temperatura del modello
        """
        self.mongodb_client = mongodb_client or MongoDBClient()
        self.embedder = embedder or Embedder()
        self.chat_model = chat_model
        self.temperature = temperature

        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)

    def ingest(
        self,
        directory: str,
        chunk_size: int = 500,
        overlap: int = 50,
        clear_existing: bool = False,
    ) -> dict:
        """
        Pipeline completa di indicizzazione: load → chunk → embed → store.

        Args:
            directory: Cartella con i documenti
            chunk_size: Token per chunk
            overlap: Token di overlap
            clear_existing: Se True, elimina i dati esistenti prima

        Returns:
            Statistiche sull'operazione
        """
        if clear_existing:
            deleted = self.mongodb_client.delete_all()
            print(f"  Eliminati {deleted} chunk esistenti")

        print(f"Caricamento documenti da: {directory}")
        documents = load_directory(directory)
        print(f"  Caricati {len(documents)} documenti")

        if not documents:
            return {"documents": 0, "chunks": 0, "status": "no_documents"}

        print("Creazione chunk...")
        chunks = chunk_documents(documents, chunk_size, overlap)
        print(f"  Creati {len(chunks)} chunk")

        print("Generazione embeddings...")
        chunks = self.embedder.embed_chunks(chunks)
        print(f"  Generati {len(chunks)} embeddings")

        print("Salvataggio in MongoDB...")
        inserted = self.mongodb_client.insert_chunks(chunks)
        print(f"  Inseriti {inserted} chunk")

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "inserted": inserted,
            "status": "success",
        }

    def query(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        return_sources: bool = True,
    ) -> dict:
        """
        Esegue una query RAG: cerca contesto e genera risposta.

        Args:
            question: La domanda dell'utente
            k: Numero di chunk da recuperare
            return_sources: Se includere le fonti nella risposta

        Returns:
            Dizionario con risposta, fonti e chunk usati
        """
        query_embedding = self.embedder.get_embedding(question)

        results = self.mongodb_client.vector_search(query_embedding, k=k)

        if not results:
            return {
                "answer": "Non ho trovato informazioni rilevanti nella knowledge base per rispondere a questa domanda.",
                "sources": [],
                "chunks": [],
            }

        prompt = self._build_prompt(question, results)

        response = self._generate_response(prompt)

        sources = []
        if return_sources:
            seen_sources = set()
            for r in results:
                source = r.get("source", "Sconosciuto")
                if source not in seen_sources:
                    sources.append(source)
                    seen_sources.add(source)

        return {
            "answer": response,
            "sources": sources,
            "chunks": results,
        }

    def _build_prompt(self, question: str, context_chunks: list[dict]) -> list[dict]:
        """
        Costruisce il prompt con contesto per il modello.

        Args:
            question: La domanda dell'utente
            context_chunks: I chunk recuperati dalla ricerca

        Returns:
            Lista di messaggi per l'API di chat
        """
        system_message = """Sei un assistente esperto che risponde alle domande basandosi ESCLUSIVAMENTE sul contesto fornito.

Regole:
1. Rispondi SOLO usando le informazioni presenti nel contesto
2. Se l'informazione non è presente nel contesto, dillo chiaramente
3. Cita le fonti quando possibile
4. Sii conciso ma completo
5. Rispondi nella stessa lingua della domanda"""

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Sconosciuto")
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            context_parts.append(f"[Fonte {i}: {source}] (relevance: {score:.2f})\n{text}")

        context_text = "\n\n---\n\n".join(context_parts)

        user_message = f"""Contesto:
{context_text}

---

Domanda: {question}"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _generate_response(self, messages: list[dict]) -> str:
        """
        Genera la risposta usando il modello di chat.

        Args:
            messages: Lista di messaggi (system + user)

        Returns:
            Risposta generata dal modello
        """
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def get_stats(self) -> dict:
        """Restituisce statistiche del database."""
        return self.mongodb_client.get_stats()

    def clear(self) -> int:
        """Elimina tutti i dati dal database."""
        return self.mongodb_client.delete_all()
