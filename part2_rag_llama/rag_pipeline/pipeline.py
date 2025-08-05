# rag_pipeline/pipeline.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_community.llms import Ollama
from .retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        csv_path: str,
        csv_column: str = "text",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chromadb",
    ):
        """
        Initialize the full RAG pipeline.
        """
        self.csv_path = csv_path
        self.csv_column = csv_column
        self.persist_directory = persist_directory

        # 1. Set up embeddings and retriever
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.retriever_tool = Retriever(self.embedding_function, persist_directory)

        # 2. Load and prepare documents
        documents = self.retriever_tool.load_documents_from_csv(csv_path, column=csv_column)
        chunks = self.retriever_tool.split_documents(documents)

        # 3. Build vector store and retriever
        self.vectorstore = self.retriever_tool.build_vectorstore(chunks)
        self.retriever = self.retriever_tool.get_retriever(self.vectorstore)

        # 4. Load local LLM via Ollama
        self.llm: BaseLLM = Ollama(model="llama2")

        # 5. Setup QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)

    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG pipeline and return the answer.
        """
        result = self.qa_chain.run(question)
        return result
