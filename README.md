# RAG Assistant

Sistema RAG (Retrieval-Augmented Generation) per interrogare documenti utilizzando intelligenza artificiale. Permette di caricare documenti, indicizzarli in un database vettoriale e fare domande ricevendo risposte contestualizzate.

## Caratteristiche

- **Elaborazione documenti**: Supporto per TXT, PDF, Markdown e JSON
- **Chunking intelligente**: Divisione automatica dei documenti in frammenti ottimizzati
- **Ricerca semantica**: Embedding vettoriali con OpenAI per recupero preciso
- **Generazione risposte**: Integrazione con GPT-4o-mini per risposte contestualizzate
- **Due interfacce**: CLI per automazione e Web UI moderna con Material Design 3
- **Database vettoriale**: MongoDB con ricerca vettoriale nativa
- **Deployment Docker**: Containerizzazione completa con Docker Compose

## Architettura

```
┌─────────────┐
│  Documenti  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Loader + Chunker│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  OpenAI Embedder│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  MongoDB Atlas  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  RAG Pipeline   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  GPT-4o-mini    │
└─────────────────┘
```

## Prerequisiti

- Python 3.8+
- **MongoDB** (scegli una delle due opzioni):
  - MongoDB locale + MongoDB Compass (consigliato per sviluppo)
  - MongoDB Atlas (cloud, consigliato per produzione)
- OpenAI API Key
- Docker e Docker Compose (opzionale, per deployment containerizzato)

## Installazione

### Opzione 1: Installazione Locale

1. Clona il repository:
```bash
git clone <repository-url>
cd rag
```

2. Crea un ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

### Opzione 2: Docker Compose

```bash
docker-compose up -d
```

## Configurazione

1. Copia il file di esempio delle variabili d'ambiente:
```bash
cp .env.example .env
```

2. Modifica `.env` con le tue credenziali:
```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# MongoDB Connection String (scegli una delle due opzioni)
# Opzione 1 - Locale (con Docker Compose):
MONGODB_URI=mongodb://localhost:27020/
# Opzione 2 - Atlas (Cloud):
# MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/

# Configurazione RAG (opzionale)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
TEMPERATURE=0.3
```

### Ottenere le credenziali

**OpenAI API Key:**
1. Vai su https://platform.openai.com/api-keys
2. Crea una nuova API key
3. Copia la chiave nel file `.env`

**MongoDB - Opzione 1: Locale con Docker Compose (Consigliato per iniziare)**
1. Usa `docker-compose up -d` per avviare MongoDB in locale
2. MongoDB sarà disponibile su `localhost:27020`
3. Usa MongoDB Compass per visualizzare i dati:
   - Scarica Compass da https://www.mongodb.com/products/compass
   - Connettiti a `mongodb://localhost:27020/`

**MongoDB - Opzione 2: Atlas (Cloud)**
1. Crea un account su https://www.mongodb.com/cloud/atlas
2. Crea un cluster gratuito
3. Ottieni la connection string
4. Sostituisci username e password nella stringa di connessione

## Utilizzo

### Interfaccia Web (Streamlit)

Avvia l'applicazione web:

```bash
streamlit run app.py
```

L'interfaccia sarà disponibile su `http://localhost:8501`

**Funzionalità:**
- Carica documenti tramite drag & drop
- Indicizza i documenti con configurazione personalizzata
- Fai domande in chat interattiva
- Visualizza fonti delle risposte
- Monitora statistiche in tempo reale

### Interfaccia CLI

#### 1. Indicizzare documenti

Indicizza tutti i documenti nella cartella `knowledge_base/`:
```bash
python main.py ingest
```

Indicizza da una cartella specifica:
```bash
python main.py ingest -d ./documenti
```

Sostituisci i dati esistenti:
```bash
python main.py ingest --clear
```

Personalizza chunking:
```bash
python main.py ingest --chunk-size 1000 --overlap 100
```

#### 2. Fare domande

Modalità interattiva:
```bash
python main.py query
```

Con configurazione custom:
```bash
python main.py query -k 10
```

#### 3. Visualizzare statistiche

```bash
python main.py stats
```

Output:
```
==================================================
STATISTICHE DATABASE
==================================================
  Database: rag_db
  Collection: embeddings
  Chunk totali: 150
  Documenti sorgente: 5
```

#### 4. Pulire il database

```bash
python main.py clear
```

Con conferma automatica:
```bash
python main.py clear --force
```

## Struttura del Progetto

```
rag/
├── src/
│   ├── __init__.py
│   ├── loaders.py          # Caricamento documenti (TXT, PDF, MD, JSON)
│   ├── chunker.py          # Divisione documenti in chunk
│   ├── embedder.py         # Generazione embeddings OpenAI
│   ├── mongodb_client.py   # Client MongoDB con ricerca vettoriale
│   └── rag_pipeline.py     # Pipeline completa RAG
├── knowledge_base/         # Documenti da indicizzare
├── main.py                 # CLI principale
├── app.py                  # Web UI Streamlit
├── setup_test_doc.py       # Script per setup iniziale
├── Dockerfile              # Containerizzazione app
├── docker-compose.yml      # Orchestrazione servizi
├── .env.example            # Template variabili d'ambiente
├── .dockerignore
├── .gitignore
└── README.md
```

## Tecnologie Utilizzate

- **Python 3.8+**: Linguaggio principale
- **OpenAI API**: Embeddings (text-embedding-3-small) e chat (GPT-4o-mini)
- **MongoDB**: Database vettoriale con ricerca semantica
- **Streamlit**: Framework per Web UI
- **Docker**: Containerizzazione e deployment
- **Python-dotenv**: Gestione variabili d'ambiente
- **tiktoken**: Tokenizzazione per OpenAI

## Parametri Configurabili

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Numero di token per chunk |
| `CHUNK_OVERLAP` | 50 | Token di sovrapposizione tra chunk |
| `TOP_K` | 5 | Numero di chunk recuperati per query |
| `EMBEDDING_MODEL` | text-embedding-3-small | Modello OpenAI per embeddings |
| `CHAT_MODEL` | gpt-4o-mini | Modello OpenAI per generazione risposte |
| `TEMPERATURE` | 0.3 | Creatività delle risposte (0.0-1.0) |

## Docker Deployment

Il progetto include configurazione Docker Compose con:
- MongoDB 7.0 in container
- Applicazione RAG containerizzata
- Network isolato per comunicazione tra servizi
- Volume persistente per dati MongoDB

Avvia tutti i servizi:
```bash
docker-compose up -d
```

Visualizza logs:
```bash
docker-compose logs -f
```

Ferma i servizi:
```bash
docker-compose down
```

Rimuovi anche i volumi:
```bash
docker-compose down -v
```

## Supporto Formati Documenti

- **TXT**: File di testo semplice
- **Markdown**: File .md e .markdown
- **PDF**: Documenti PDF
- **JSON**: File JSON strutturati

## Troubleshooting

### Errore di connessione MongoDB (Locale)
- Verifica che Docker Compose sia in esecuzione: `docker-compose ps`
- Controlla che la porta 27020 non sia in uso: `netstat -an | findstr 27020`
- Riavvia il container: `docker-compose restart mongodb`
- Verifica i logs: `docker-compose logs mongodb`

### Errore di connessione MongoDB (Atlas)
- Verifica che la connection string in `.env` sia corretta
- Controlla che l'IP sia whitelistato su MongoDB Atlas
- Verifica le credenziali username/password

### Errore OpenAI API
- Controlla che l'API key sia valida
- Verifica di avere crediti disponibili
- Controlla i rate limits

### Errore chunking
- Riduci `CHUNK_SIZE` se i documenti sono molto grandi
- Aumenta `CHUNK_OVERLAP` per migliorare il contesto

## Roadmap

- [ ] Supporto per più formati (DOCX, CSV, Excel)
- [ ] Integrazione con altri LLM (Anthropic, Cohere)
- [ ] Cache delle risposte
- [ ] Modalità streaming per risposte lunghe
- [ ] Export conversazioni
- [ ] Multi-tenancy e autenticazione

## License

Questo progetto è fornito as-is per scopi educativi e di sviluppo.

## Contatti

Per domande o supporto, apri una issue nel repository.
