import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document


class Retriever:
    def __init__(self, embedding_function, persist_directory: str = "./chromadb"):
        """
        Initialize the Retriever with embedding and Chroma vectorstore.
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory

    def load_documents_from_csv(self, csv_path: str, column: str = "text"):
        """
        Load and convert a CSV column into LangChain Document objects.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        return [Document(page_content=row[column]) for _, row in df.iterrows() if pd.notna(row[column])]

    def split_documents(self, documents, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Split documents into smaller chunks for vector storage.
        """
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)

    def build_vectorstore(self, documents):
        """
        Create and persist a Chroma vector store.
        """
        vectorstore = Chroma.from_documents(
            documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        vectorstore.persist()
        return vectorstore

    def get_retriever(self, vectorstore):
        """
        Return a retriever instance from the Chroma vectorstore.
        """
        return vectorstore.as_retriever()
