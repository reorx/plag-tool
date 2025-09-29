"""Core modules for plagiarism detection."""

from .config import Config
from .chunker import TextChunk, TextChunker
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .detector import PlagiarismDetector, PlagiarismReport, Match
from .report import ReportGenerator

__all__ = [
    "Config",
    "TextChunk",
    "TextChunker",
    "EmbeddingService",
    "VectorStore",
    "PlagiarismDetector",
    "PlagiarismReport",
    "Match",
    "ReportGenerator",
]