from pathlib import Path

from ingester import build_default_ingester
from rag_pipeline import build_default_pipeline


def main():
    """Entry point: run ingestion then start the interactive RAG assistant."""

    # Build and run the ingestion pipeline
    ingester = build_default_ingester(data_dir=Path("data"))
    ingester.run()

    # Build the RAG query pipeline
    pipeline = build_default_pipeline()

    print("RAG assistant is ready.")
    print("Ask a question about Fantasy Realms RPG.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        answer = pipeline.ask(query)
        print("\nAnswer:")
        print(answer)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
