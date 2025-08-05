# create_vectorstore.py

from rag_pipeline.retriever import Retriever
from langchain_huggingface import HuggingFaceEmbeddings

CSV_PATH = "data/f1Data/results.csv"
COLUMN = "text"
PERSIST_DIR = "./chromadb"

if __name__ == "__main__":
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_function, persist_directory=PERSIST_DIR)

    documents = retriever.load_documents_from_csv(CSV_PATH, column=COLUMN)
    chunks = retriever.split_documents(documents)
    vectorstore = retriever.build_vectorstore(chunks)
    vectorstore.persist()

    print("✅ Vector store created and saved to", PERSIST_DIR)
