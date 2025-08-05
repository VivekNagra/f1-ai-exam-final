# cli.py
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from rag_pipeline.pipeline import RAGPipeline

import sys
from rag_pipeline.pipeline import RAGPipeline

def main():
    if len(sys.argv) < 2:
        print("Please provide a question as an argument.")
        print("Usage: python cli.py 'Your question here'")
        sys.exit(1)

    question = sys.argv[1]


    pipeline = RAGPipeline(
        csv_path="../data/f1Data/results.csv",
        persist_directory="./chromadb"
    )

    answer = pipeline.run(question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
