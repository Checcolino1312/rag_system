"""
Script per inserire un documento di test in MongoDB Atlas.
Questo permette di creare l'indice vettoriale.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Connessione con timeout esteso
uri = os.getenv("MONGODB_URI")
print(f"Connessione a MongoDB Atlas...")
client = MongoClient(uri, serverSelectionTimeoutMS=60000, connectTimeoutMS=60000)

# Crea database e collection
db = client["rag_db"]
collection = db["chunks"]

# Documento di test con embedding finto (1536 dimensioni)
test_doc = {
    "chunk_id": "test-setup-001",
    "text": "Questo è un documento di test per configurare l'indice vettoriale.",
    "source": "setup_test",
    "metadata": {"type": "test", "chunk_index": 0, "total_chunks": 1},
    "embedding": [0.0] * 1536  # Vettore di 1536 zeri
}

# Inserisci documento
result = collection.insert_one(test_doc)
print(f"Documento inserito con ID: {result.inserted_id}")
print(f"Database: rag_db")
print(f"Collection: chunks")
print()
print("Ora puoi creare l'indice vettoriale su Atlas!")
print("Vai su Atlas Search e usa questi valori:")
print("  - Index Name: vector_index")
print("  - Database: rag_db")
print("  - Collection: chunks")

client.close()
