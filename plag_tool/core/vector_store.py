"""Vector store module using ChromaDB for similarity search."""

import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .chunker import TextChunk
from .log import base_logger

logger = base_logger.getChild('vector_store')


class SimilarityMatch:
    """Represents a similarity search result."""

    def __init__(self, chunk_id: str, similarity: float, metadata: Dict[str, Any]):
        """Initialize a similarity match."""
        self.chunk_id = chunk_id
        self.similarity = similarity
        self.metadata = metadata
        self.text = metadata.get("text", "")
        self.start_pos = metadata.get("start_pos", 0)
        self.end_pos = metadata.get("end_pos", 0)
        self.doc_id = metadata.get("doc_id", "")

    def __repr__(self):
        """String representation."""
        return f"SimilarityMatch(id={self.chunk_id}, similarity={self.similarity:.3f}, doc={self.doc_id})"


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory for ChromaDB persistence
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = None
        self.collection_name = None

    def create_collection(self, name: Optional[str] = None, reset: bool = True):
        """
        Create or get a collection for storing vectors.

        Args:
            name: Collection name (auto-generated if not provided)
            reset: Whether to delete existing collection with same name
        """
        if name is None:
            name = f"plagiarism_{uuid.uuid4().hex[:8]}"

        self.collection_name = name

        if reset:
            try:
                self.client.delete_collection(name)
                logger.debug(f"Deleted existing collection: {name}")
            except Exception:
                pass  # Collection doesn't exist

        try:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"description": "Plagiarism detection vectors"}
            )
            logger.info(f"Created collection: {name}")
        except Exception as e:
            # Collection already exists, get it
            self.collection = self.client.get_collection(name)
            logger.info(f"Using existing collection: {name}")

    def add_documents(self, chunks: List[TextChunk], embeddings: List[np.ndarray]):
        """
        Store document chunks with their embeddings.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            logger.warning("No chunks to add to vector store")
            return

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        embeddings_list = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{chunk.doc_id}_{chunk.chunk_index}_{chunk.chunk_hash[:8]}"
            ids.append(chunk_id)

            metadata = {
                "text": chunk.text,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "chunk_hash": chunk.chunk_hash
            }
            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise

    def search_similar(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.85,
        top_k: int = 10,
        filter_doc_id: Optional[str] = None
    ) -> List[SimilarityMatch]:
        """
        Find similar chunks based on embedding similarity.

        Args:
            query_embedding: Query embedding vector
            threshold: Minimum similarity threshold (0-1)
            top_k: Maximum number of results to return
            filter_doc_id: Optional document ID to filter results

        Returns:
            List of SimilarityMatch objects
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        # Prepare query
        where_clause = None
        if filter_doc_id:
            where_clause = {"doc_id": {"$ne": filter_doc_id}}

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )

        # Process results
        matches = []
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for idx in range(len(results['ids'][0])):
                # ChromaDB returns distance, convert to similarity
                # Using cosine distance: similarity = 1 - distance
                distance = results['distances'][0][idx]
                similarity = 1 - distance

                if similarity >= threshold:
                    match = SimilarityMatch(
                        chunk_id=results['ids'][0][idx],
                        similarity=similarity,
                        metadata=results['metadatas'][0][idx]
                    )
                    matches.append(match)

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x.similarity, reverse=True)

        return matches

    def batch_search_similar(
        self,
        query_embeddings: List[np.ndarray],
        threshold: float = 0.85,
        top_k: int = 10
    ) -> List[List[SimilarityMatch]]:
        """
        Perform batch similarity search for multiple queries.

        Args:
            query_embeddings: List of query embedding vectors
            threshold: Minimum similarity threshold
            top_k: Maximum number of results per query

        Returns:
            List of lists containing SimilarityMatch objects
        """
        all_matches = []
        for embedding in query_embeddings:
            matches = self.search_similar(embedding, threshold, top_k)
            all_matches.append(matches)
        return all_matches

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {"error": "Collection not initialized"}

        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_dir
        }

    def delete_collection(self):
        """Delete the current collection."""
        if self.collection and self.collection_name:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                self.collection = None
                self.collection_name = None
            except Exception as e:
                logger.error(f"Failed to delete collection: {str(e)}")
                raise

    def list_collections(self) -> List[str]:
        """
        List all collections in the database.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def clear_collection(self):
        """Clear all documents from the current collection."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        # Get all IDs and delete them
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)
            logger.info(f"Cleared {len(all_ids)} documents from collection")

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            List of chunk metadata dictionaries
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        results = self.collection.get(
            where={"doc_id": doc_id}
        )

        chunks = []
        if results and results['ids']:
            for i in range(len(results['ids'])):
                chunk_data = {
                    "id": results['ids'][i],
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                }
                chunks.append(chunk_data)

        # Sort by chunk_index
        chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))

        return chunks