"""Shared data types and models for the plagiarism detection system."""

import hashlib
from typing import List, Dict, Any
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


class Match(BaseModel):
    """Represents a plagiarism match between source and target texts."""

    source_text: str = Field(description="Source text segment")
    source_start: int = Field(description="Start position in source document")
    source_end: int = Field(description="End position in source document")
    target_text: str = Field(description="Target text segment")
    target_start: int = Field(description="Start position in target document")
    target_end: int = Field(description="End position in target document")
    similarity: float = Field(description="Similarity score (0-1)")
    exact_matches: List[str] = Field(default_factory=list, description="List of exact matching phrases")


class PlagiarismReport(BaseModel):
    """Complete plagiarism detection report."""

    source_file: str = Field(description="Path to source file")
    target_file: str = Field(description="Path to target file")
    total_matches: int = Field(description="Total number of matches found")
    plagiarism_percentage: float = Field(description="Percentage of source text that is plagiarized")
    matches: List[Match] = Field(default_factory=list, description="List of all matches")
    source_length: int = Field(description="Total length of source text")
    target_length: int = Field(description="Total length of target text")
    detection_threshold: float = Field(description="Similarity threshold used for detection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")