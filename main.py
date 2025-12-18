"""
RAG System - Entry Point CLI
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline


def cmd_ingest(args):
    """Comando per indicizzare i documenti."""
    print("=" * 50)
    print("INDICIZZAZIONE DOCUMENTI")
    print("=" * 50)

    pipeline = RAGPipeline()

    result = pipeline.ingest(
        directory=args.directory,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        clear_existing=args.clear,
    )

    print()
    print("Risultato:")
    print(f"  Documenti caricati: {result['documents']}")
    print(f"  Chunk creati: {result['chunks']}")
    print(f"  Chunk inseriti: {result.get('inserted', 0)}")
    print(f"  Status: {result['status']}")


def cmd_query(args):
    """Comando per fare domande in modalità interattiva."""
    print("=" * 50)
    print("MODALITÀ QUERY")
    print("=" * 50)
    print("Digita 'exit' o 'quit' per uscire")
    print()

    pipeline = RAGPipeline()

    while True:
        try:
            question = input("\nDomanda: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                print("Arrivederci!")
                break

            print("\nCerco risposta...")
            result = pipeline.query(question, k=args.top_k)

            print("\n" + "-" * 40)
            print("RISPOSTA:")
            print("-" * 40)
            print(result["answer"])

            if result["sources"]:
                print("\n" + "-" * 40)
                print("FONTI:")
                print("-" * 40)
                for source in result["sources"]:
                    print(f"  - {source}")

        except KeyboardInterrupt:
            print("\n\nInterrotto. Arrivederci!")
            break
        except Exception as e:
            print(f"\nErrore: {e}")


def cmd_stats(args):
    """Comando per visualizzare le statistiche."""
    print("=" * 50)
    print("STATISTICHE DATABASE")
    print("=" * 50)

    pipeline = RAGPipeline()
    stats = pipeline.get_stats()

    print(f"  Database: {stats['database']}")
    print(f"  Collection: {stats['collection']}")
    print(f"  Chunk totali: {stats['total_chunks']}")
    print(f"  Documenti sorgente: {stats['total_documents']}")


def cmd_clear(args):
    """Comando per pulire il database."""
    if not args.force:
        confirm = input("Sei sicuro di voler eliminare tutti i dati? (s/N): ")
        if confirm.lower() != "s":
            print("Operazione annullata.")
            return

    print("Eliminazione dati in corso...")

    pipeline = RAGPipeline()
    deleted = pipeline.clear()

    print(f"Eliminati {deleted} chunk.")


def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG - Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main.py ingest                    # Indicizza documenti da knowledge_base/
  python main.py ingest -d ./docs          # Indicizza da cartella specifica
  python main.py query                     # Modalità domande interattiva
  python main.py stats                     # Mostra statistiche
  python main.py clear                     # Pulisce il database
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # Comando ingest
    parser_ingest = subparsers.add_parser("ingest", help="Indicizza documenti nella knowledge base")
    parser_ingest.add_argument(
        "-d",
        "--directory",
        default="knowledge_base",
        help="Cartella con i documenti (default: knowledge_base)",
    )
    parser_ingest.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "500")),
        help="Token per chunk (default: 500)",
    )
    parser_ingest.add_argument(
        "--overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "50")),
        help="Token di overlap (default: 50)",
    )
    parser_ingest.add_argument(
        "--clear",
        action="store_true",
        help="Elimina dati esistenti prima di indicizzare",
    )
    parser_ingest.set_defaults(func=cmd_ingest)

    # Comando query
    parser_query = subparsers.add_parser("query", help="Fai domande in modalità interattiva")
    parser_query.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", "5")),
        help="Numero di chunk da recuperare (default: 5)",
    )
    parser_query.set_defaults(func=cmd_query)

    # Comando stats
    parser_stats = subparsers.add_parser("stats", help="Mostra statistiche del database")
    parser_stats.set_defaults(func=cmd_stats)

    # Comando clear
    parser_clear = subparsers.add_parser("clear", help="Elimina tutti i dati dal database")
    parser_clear.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Non chiedere conferma",
    )
    parser_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"\nErrore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
