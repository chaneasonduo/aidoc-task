import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, splitting, and vector storage."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
        )
        self.vector_store_path = settings.VECTOR_STORE_PATH
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def get_loader(self, file_path: str):
        """Get the appropriate document loader based on file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return PyPDFLoader(file_path)
        elif file_ext in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path)
        elif file_ext in ['.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif file_ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and split a document into chunks."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Get the appropriate loader
            loader = self.get_loader(file_path)
            
            # Load and split the document
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Convert Document objects to dictionaries
            chunks_dict = [
                {
                    "page_content": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "source": str(chunk.metadata.get("source", file_path)),
                    },
                }
                for chunk in chunks
            ]
            
            return chunks_dict
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def store_documents(self, documents: List[Dict[str, Any]], collection_name: str = "documents") -> None:
        """Store document chunks in the vector database."""
        try:
            # Create Chroma vector store
            db = Chroma.from_documents(
                documents=[self._dict_to_document(doc) for doc in documents],
                embedding=self.embeddings,
                persist_directory=self.vector_store_path,
                collection_name=collection_name,
            )
            
            # Persist the database
            db.persist()
            logger.info(f"Stored {len(documents)} document chunks in collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise
    
    def search_documents(self, query: str, collection_name: str = "documents", k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents to the query."""
        try:
            # Load the vector store
            db = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
            
            # Search for similar documents
            results = db.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def _dict_to_document(self, doc_dict: Dict[str, Any]) -> Dict:
        """Convert a dictionary to a document format that Chroma can understand."""
        from langchain.schema import Document
        return Document(
            page_content=doc_dict["page_content"],
            metadata=doc_dict.get("metadata", {})
        )
    
    def get_collections(self) -> List[str]:
        """Get a list of all collections in the vector store."""
        try:
            # Initialize Chroma without a specific collection
            db = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            
            # Get all collections
            collections = db._client.list_collections()
            return [collection.name for collection in collections]
            
        except Exception as e:
            logger.error(f"Error retrieving collections: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from the vector store."""
        try:
            # Initialize Chroma with the target collection
            db = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # Delete the collection
            db.delete_collection()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            return False
