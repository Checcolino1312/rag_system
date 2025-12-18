# Costruire un RAG da Zero

## Guida Progettuale Passo Passo

---

## Introduzione: Cosa Stiamo Costruendo

Prima di scrivere una sola riga di codice, è fondamentale capire cosa vogliamo ottenere. Un sistema RAG (Retrieval-Augmented Generation) risolve un problema specifico: i modelli linguistici come GPT hanno una conoscenza "congelata" al momento del loro addestramento e non sanno nulla dei tuoi documenti specifici.

L'idea è semplice ma potente: invece di chiedere al modello di rispondere solo con la sua conoscenza generale, prima recuperiamo i pezzi più rilevanti dalla nostra base documentale, e poi li passiamo al modello come contesto. È come dare a qualcuno i capitoli giusti di un libro prima di fargli una domanda su quel libro.

Il flusso che costruiremo funziona così: l'utente fa una domanda, il sistema cerca nei documenti i passaggi più pertinenti, questi vengono inseriti nel prompt insieme alla domanda, e il modello genera una risposta basata su quel contesto specifico.

---

## Fase 1: Preparare la Knowledge Base

### Raccogliere i Documenti

Il primo passo è decidere quali documenti costituiranno la tua base di conoscenza. Possono essere manuali tecnici, FAQ, documentazione di prodotto, articoli, policy aziendali, qualsiasi testo che contenga le informazioni che vuoi rendere accessibili.

Crea una cartella dedicata, ad esempio `knowledge_base/`, e organizza i documenti in modo logico. Non preoccuparti troppo del formato: potrai gestire PDF, file di testo, Markdown e JSON. L'importante è che il contenuto sia testuale e di qualità.

**Consiglio pratico:** inizia con pochi documenti (5-10) per testare l'intero flusso, poi scala gradualmente. È molto più facile debuggare problemi su un dataset piccolo.

### Creare i Document Loader

Dovrai scrivere funzioni che leggono ogni tipo di file e ne estraggono il testo. Per i file TXT e Markdown è banale: basta leggere il contenuto. Per i PDF userai una libreria come PyPDF2 che estrae il testo da ogni pagina.

Ogni documento caricato dovrebbe diventare un dizionario con almeno tre campi: il contenuto testuale, il percorso del file sorgente (utile per le citazioni), e il tipo di documento. Questa struttura ti permetterà di tracciare da dove viene ogni informazione.

---

## Fase 2: Il Chunking dei Documenti

### Perché Dividere in Chunk

Qui arriviamo a un punto cruciale. Non puoi semplicemente passare interi documenti al modello: alcuni sono troppo lunghi, e soprattutto la ricerca semantica funziona meglio su porzioni di testo più piccole e focalizzate.

Il chunking è l'arte di dividere i documenti in segmenti di dimensione appropriata. Troppo piccoli e perdi il contesto; troppo grandi e la ricerca diventa imprecisa. Un buon punto di partenza sono chunk di 400-500 token.

### La Strategia dell'Overlap

C'è un trucco importante: i chunk non dovrebbero essere completamente separati. Se tagli un documento ogni 500 token esatti, rischi di spezzare concetti a metà. La soluzione è l'overlap: ogni chunk condivide una porzione (tipicamente 50-100 token) con quello precedente.

Immagina di leggere un libro con un segnalibro che copre sempre le ultime due righe della pagina precedente: questo è l'overlap. Garantisce continuità semantica tra i chunk.

### Implementazione del Chunker

Per implementare il chunker userai la libreria `tiktoken`, che è il tokenizzatore ufficiale di OpenAI. Ti permette di contare esattamente quanti token occupa un testo, così puoi fare tagli precisi.

Il flusso è: prendi il testo, lo converti in token, scorri con una finestra della dimensione desiderata, converti ogni finestra di token in testo, e salvi ogni chunk con un identificatore univoco e il riferimento al documento sorgente.

---

## Fase 3: Generare gli Embeddings

### Cosa Sono gli Embeddings

Gli embeddings sono il cuore della ricerca semantica. Un embedding è una rappresentazione numerica del significato di un testo: un vettore di numeri (tipicamente 1536 dimensioni per i modelli OpenAI) che cattura il "senso" di quel testo.

La magia è che testi con significato simile avranno embeddings simili, cioè vettori che puntano in direzioni vicine nello spazio multidimensionale. "Il gatto dorme" e "Il felino riposa" avranno embeddings molto vicini, anche se le parole sono diverse.

### Scegliere il Modello

OpenAI offre diversi modelli di embedding. Per iniziare, `text-embedding-3-small` è perfetto: costa poco, è veloce, e produce vettori di qualità. Se in futuro avrai bisogno di maggiore precisione, potrai passare a `text-embedding-3-large`.

**Nota sui costi:** gli embeddings sono molto economici. Puoi processare migliaia di chunk spendendo pochi centesimi. Il costo principale sarà nelle chiamate al modello di chat per generare le risposte.

### Il Processo di Embedding

Per ogni chunk che hai creato, farai una chiamata API a OpenAI per ottenere il suo embedding. La risposta sarà un array di 1536 numeri decimali. Questo array va salvato insieme al chunk nel database.

Considera di implementare il batching: invece di fare una chiamata per chunk, puoi inviare più testi insieme. OpenAI supporta fino a 2048 input per chiamata, il che velocizza enormemente il processo per grandi knowledge base.

---

## Fase 4: Configurare MongoDB Atlas

### Perché MongoDB con Vector Search

Hai bisogno di un database che sappia fare due cose: memorizzare i tuoi chunk con i loro embeddings, e cercare velocemente quali embeddings sono più simili a un vettore di query. MongoDB Atlas con Vector Search fa esattamente questo.

La scelta di MongoDB è pragmatica: il tier gratuito è generoso, l'interfaccia è intuitiva, e non devi gestire infrastruttura. Per un progetto di apprendimento o un MVP, è perfetto. Alternative come Pinecone o Weaviate sono valide ma aggiungono complessità.

### Setup del Cluster

Vai su mongodb.com/atlas e crea un account. Scegli il tier M0 (gratuito) e una regione geograficamente vicina a te. Durante la configurazione ti verrà chiesto di creare un utente database e di specificare quali IP possono connettersi.

**Per lo sviluppo:** puoi permettere connessioni da qualsiasi IP (0.0.0.0/0), ma in produzione dovresti limitare agli IP dei tuoi server.

### Creare l'Indice Vettoriale

Questo è un passaggio che molti dimenticano: devi creare esplicitamente un indice di tipo "vector" sul campo che conterrà gli embeddings. Senza questo indice, le query vettoriali non funzioneranno.

Vai nella sezione Atlas Search del tuo cluster, crea un nuovo indice, scegli JSON Editor, e definisci un indice sul campo "embedding" specificando che è un vettore di 1536 dimensioni con similarità coseno. Dai all'indice un nome come `vector_index` che userai nel codice.

La configurazione JSON dell'indice sarà simile a questa:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

### Struttura dei Documenti

Ogni documento nella collection avrà questa struttura: un `chunk_id` univoco, il testo del chunk, l'embedding (l'array di 1536 numeri), il percorso del documento sorgente, e opzionalmente metadati aggiuntivi come data di creazione o categoria.

---

## Fase 5: Costruire la Pipeline RAG

### Il Flusso Completo

Ora hai tutti i pezzi. La pipeline RAG li orchestra così: arriva una domanda dall'utente, generi l'embedding della domanda usando lo stesso modello usato per i chunk, cerchi nel database i chunk con embeddings più simili, costruisci un prompt che include questi chunk come contesto, e chiami il modello di chat per generare la risposta.

### La Ricerca Semantica

Quando l'utente chiede "Come configuro l'autenticazione?", generi l'embedding di questa domanda. Poi usi l'operatore `$vectorSearch` di MongoDB per trovare i chunk i cui embeddings sono più vicini (in termini di similarità coseno) all'embedding della domanda.

Tipicamente recuperi i top 3-5 chunk più rilevanti. Questo parametro (k) è configurabile: più chunk significa più contesto ma anche più token consumati e potenzialmente più rumore.

### Costruire il Prompt

Il prompt che invii al modello di chat ha una struttura precisa. Nel messaggio di sistema spieghi che deve rispondere basandosi solo sul contesto fornito, e che deve ammettere se non trova l'informazione. Nel messaggio utente includi prima il contesto (i chunk recuperati, con le loro fonti) e poi la domanda.

**Esempio di struttura:**

```
Sistema: "Sei un assistente. Rispondi basandoti SOLO sul contesto seguente. 
Se l'informazione non è presente, dillo chiaramente."

Utente: "Contesto:
[Documento 1 - manuale.pdf]: ...testo del chunk...
[Documento 2 - faq.md]: ...testo del chunk...

Domanda: Come configuro l'autenticazione?"
```

### Scegliere il Modello di Chat

Per la generazione delle risposte, `gpt-4o-mini` è un ottimo compromesso tra qualità e costo. È veloce, economico, e sufficientemente capace di sintetizzare informazioni dal contesto. Per casi d'uso più critici, puoi passare a `gpt-4o`.

---

## Fase 6: Testing e Iterazione

### Testare con Domande Reali

Una volta assemblata la pipeline, testala con domande che sai avere risposta nei tuoi documenti. Verifica che i chunk recuperati siano effettivamente pertinenti. Se non lo sono, potresti dover rivedere la dimensione dei chunk o la qualità dei documenti sorgente.

Fai anche test negativi: domande su argomenti non presenti nella knowledge base. Il sistema dovrebbe ammettere di non avere l'informazione, non inventare.

### Parametri da Ottimizzare

I parametri chiave da sperimentare sono:

| Parametro | Range consigliato | Note |
|-----------|-------------------|------|
| Chunk size | 300-800 token | Dipende dalla natura dei documenti |
| Overlap | 50-150 token | ~10-20% del chunk size |
| Top-k risultati | 3-7 | Più alto = più contesto ma più rumore |
| Temperatura LLM | 0.3-0.7 | Più bassa = più deterministico |

Non esiste una configurazione universale: dipende dai tuoi documenti e casi d'uso.

### Debugging Comune

Se le risposte sono imprecise, prima verifica che i chunk recuperati siano sensati. Se i chunk sono giusti ma la risposta è sbagliata, il problema è nel prompt o nel modello. Se i chunk sono sbagliati, il problema è nel chunking o negli embeddings.

---

## Fase 7: Struttura del Progetto

Organizza il codice in moduli separati. Una struttura consigliata:

```
rag-project/
├── knowledge_base/          # I tuoi documenti sorgente
│   ├── manuale.pdf
│   ├── faq.md
│   └── ...
├── src/
│   ├── loaders.py           # Caricamento documenti
│   ├── chunker.py           # Logica di chunking
│   ├── embedder.py          # Generazione embeddings
│   ├── mongodb_client.py    # Interazione con MongoDB
│   └── rag_pipeline.py      # Orchestrazione completa
├── .env                     # Credenziali (OPENAI_API_KEY, MONGODB_URI)
├── requirements.txt         # Dipendenze Python
└── main.py                  # Entry point
```

Questa separazione ti permette di testare ogni componente indipendentemente e di sostituire parti senza riscrivere tutto. Per esempio, potresti voler cambiare database in futuro senza toccare la logica di chunking.

---

## Prossimi Passi

Una volta che il sistema base funziona, ci sono molte direzioni per migliorarlo:

- Aggiungere un'interfaccia web con Streamlit o Gradio
- Implementare il re-ranking dei risultati
- Aggiungere filtri sui metadati
- Gestire aggiornamenti incrementali della knowledge base
- Integrare memoria conversazionale per chat multi-turno
- Implementare hybrid search (vettoriale + keyword)

Ma il consiglio più importante è: **inizia semplice**. Fai funzionare la versione base, usala davvero, e lascia che siano i problemi reali a guidare le ottimizzazioni. Un RAG funzionante e imperfetto vale più di un RAG perfetto mai completato.

---

*Buon lavoro!*
