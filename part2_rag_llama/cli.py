import sys
from .rag_pipeline.pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print("Please provide a question as an argument.")
        print("Usage: python main.py 'Your question here'")
        sys.exit(1)

    question = sys.argv[1]
    pipeline = RAGPipeline(
        csv_path="data/f1Data/results.csv",
        csv_column="text",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chromadb"
    )

    answer = pipeline.ask(question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
