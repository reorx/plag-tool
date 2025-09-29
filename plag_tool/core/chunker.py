"""Text chunking module for splitting documents into overlapping segments."""

import hashlib
from typing import List, Optional
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Represents a chunk of text with metadata."""

    text: str = Field(description="The actual text content of the chunk")
    start_pos: int = Field(description="Starting position in the original text")
    end_pos: int = Field(description="Ending position in the original text")
    doc_id: str = Field(description="Identifier of the source document")
    chunk_index: int = Field(description="Index of this chunk in the document")
    chunk_hash: str = Field(default="", description="Hash of the chunk text")

    def __init__(self, **data):
        """Initialize a TextChunk and compute its hash."""
        super().__init__(**data)
        if not self.chunk_hash:
            self.chunk_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of the chunk text."""
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:16]

    def overlaps_with(self, other: 'TextChunk', tolerance: int = 0) -> bool:
        """Check if this chunk overlaps with another chunk."""
        if self.doc_id != other.doc_id:
            return False
        return (
            (self.start_pos <= other.start_pos <= self.end_pos + tolerance) or
            (other.start_pos <= self.start_pos <= other.end_pos + tolerance)
        )


class TextChunker:
    """Handles text segmentation with sliding window approach."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

    def chunk_text(self, text: str, doc_id: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk
            doc_id: Identifier for the document

        Returns:
            List of TextChunk objects
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk text
            chunk_text = text[start:end]

            # Create chunk object
            chunk = TextChunk(
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                doc_id=doc_id,
                chunk_index=chunk_index
            )
            chunks.append(chunk)

            # Move to next chunk
            start += self.stride
            chunk_index += 1

            # Break if we've reached the end
            if end >= text_length:
                break

        return chunks

    def chunk_with_sentences(self, text: str, doc_id: str) -> List[TextChunk]:
        """
        Split text into chunks, trying to preserve sentence boundaries.
        This is especially useful for Chinese text.

        Args:
            text: The text to chunk
            doc_id: Identifier for the document

        Returns:
            List of TextChunk objects
        """
        if not text:
            return []

        # Common sentence delimiters for Chinese and English
        sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';', '\n\n']

        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            # Calculate ideal end position
            ideal_end = min(start + self.chunk_size, text_length)

            # Look for sentence boundary near the ideal end
            end = ideal_end
            if ideal_end < text_length:
                # Search for sentence ending within a window
                search_start = max(ideal_end - 50, start + self.chunk_size // 2)
                search_end = min(ideal_end + 50, text_length)

                best_pos = ideal_end
                for pos in range(search_start, search_end):
                    if pos < text_length and text[pos] in sentence_endings:
                        best_pos = pos + 1
                        break

                end = best_pos

            # Extract chunk text
            chunk_text = text[start:end]

            # Create chunk object
            chunk = TextChunk(
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                doc_id=doc_id,
                chunk_index=chunk_index
            )
            chunks.append(chunk)

            # Move to next chunk
            start = end - self.overlap if end - self.overlap > start else end
            chunk_index += 1

            # Break if we've reached the end
            if end >= text_length:
                break

        return chunks

    def merge_small_chunks(self, chunks: List[TextChunk], min_size: int = 100) -> List[TextChunk]:
        """
        Merge small chunks with their neighbors.

        Args:
            chunks: List of chunks to process
            min_size: Minimum size for a chunk

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if current chunk is too small and can be merged
            if len(current.text) < min_size and current.doc_id == next_chunk.doc_id:
                # Merge with next chunk
                merged_text = current.text + next_chunk.text[self.overlap:] if self.overlap > 0 else current.text + next_chunk.text
                current = TextChunk(
                    text=merged_text,
                    start_pos=current.start_pos,
                    end_pos=next_chunk.end_pos,
                    doc_id=current.doc_id,
                    chunk_index=current.chunk_index
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged