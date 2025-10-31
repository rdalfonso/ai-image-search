"""
ChromaDB service for managing vector database operations
"""

import chromadb

from .config import Config


class ChromaDBService:
    """Manages ChromaDB operations"""
    
    def __init__(self, db_path: str = Config.CHROMA_DB_PATH, collection_name: str = Config.COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a single document to the collection"""
        try:
            self.collection.add(
                documents=[text],
                ids=[doc_id]
            )
            print(f"Added document {doc_id}")
        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")
    
    def search(self, query: str, n_results: int = Config.MAX_RESULTS) -> dict:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            
        Returns:
            Query results dictionary
        """
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()