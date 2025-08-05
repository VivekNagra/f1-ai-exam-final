# rag_pipeline/pipeline.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd

from .retriever import RetrieverTool

class RAGPipeline:
    def __init__(
        self,
        csv_path: str,
        persist_directory: str = "./chromadb",
    ):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.retriever_tool = RetrieverTool()

    
        documents = self.retriever_tool.load_documents_from_csv(csv_path)
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        split_docs = splitter.split_documents(documents)

    
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        
        self.llm = Ollama(model="llama2")

       
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever()
        )

    def run(self, query: str) -> str:
        return self.qa_chain.run(query)
