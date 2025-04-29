from langchain_community.vectorstores import Chroma
from app.config import CHROMA_DIR
from typing import List
from langchain.schema import Document
import os

class VectorStore:
    def __init__(self):
        self.persist_dir = CHROMA_DIR
        self.vector_store = None
        self.retriever = None

    def initialize(self, chunks: List[Document], embeddings):
        """Initialize or load vector store"""
        try:
            if os.path.exists(os.path.join(self.persist_dir, "chroma-collections.parquet")):
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=embeddings
                )
                print("Loaded existing vector store")
            else:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(self.persist_dir)
                )
                print("Created new vector store")

            # Initialize retriever with MMR for better diversity
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",  # Maximal Marginal Relevance
                search_kwargs={"k": 4, "lambda_mult": 0.25}
            )
            
            return self.retriever

        except Exception as e:
            print(f"Vector store error: {str(e)}")
            raise

    def get_retriever(self):
        if not self.retriever:
            raise ValueError("Vector store not initialized")
        return self.retriever

    def clear_db(self):
        """For development: Clear existing database"""
        import shutil
        shutil.rmtree(self.persist_dir)
        print(f"Cleared vector store at {self.persist_dir}")