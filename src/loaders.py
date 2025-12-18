"""
Document Loaders - Caricamento documenti da vari formati
"""

import json
import os
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader


def load_txt(path: str) -> dict:
    """Carica un file di testo."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "content": content,
        "source": path,
        "type": "txt",
    }


def load_markdown(path: str) -> dict:
    """Carica un file Markdown."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "content": content,
        "source": path,
        "type": "markdown",
    }


def load_pdf(path: str) -> dict:
    """Estrae il testo da un file PDF."""
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    content = "\n\n".join(text_parts)
    return {
        "content": content,
        "source": path,
        "type": "pdf",
    }


def load_json(path: str) -> dict:
    """
    Carica un file JSON.
    Cerca un campo 'content' o 'text', altrimenti converte tutto in stringa.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        content = data.get("content") or data.get("text") or json.dumps(data, indent=2)
    elif isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                parts.append(item.get("content") or item.get("text") or json.dumps(item))
            else:
                parts.append(str(item))
        content = "\n\n".join(parts)
    else:
        content = str(data)

    return {
        "content": content,
        "source": path,
        "type": "json",
    }


def load_document(path: str) -> Optional[dict]:
    """
    Carica un singolo documento in base alla sua estensione.
    Restituisce None se il formato non è supportato.
    """
    path_obj = Path(path)
    extension = path_obj.suffix.lower()

    loaders = {
        ".txt": load_txt,
        ".md": load_markdown,
        ".markdown": load_markdown,
        ".pdf": load_pdf,
        ".json": load_json,
    }

    loader = loaders.get(extension)
    if loader:
        try:
            return loader(path)
        except Exception as e:
            print(f"Errore caricando {path}: {e}")
            return None
    return None


def load_directory(directory: str) -> list[dict]:
    """
    Carica tutti i documenti supportati da una cartella.
    Supporta: .txt, .md, .markdown, .pdf, .json
    """
    documents = []
    supported_extensions = {".txt", ".md", ".markdown", ".pdf", ".json"}

    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory non trovata: {directory}")

    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            doc = load_document(str(file_path))
            if doc and doc["content"].strip():
                documents.append(doc)
                print(f"  Caricato: {file_path.name}")

    return documents
