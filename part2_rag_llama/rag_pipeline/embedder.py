from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFace embedding model.
        """
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def get(self):
        """
        Return the embedding function object for use in vectorstore or retriever.
        """
        return self.embedding_model
