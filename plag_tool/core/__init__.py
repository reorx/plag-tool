"""Core modules for plagiarism detection."""

from .config import Config
from .splitter import TextChunk, TextSplitter
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .detector import PlagiarismDetector, PlagiarismReport, Match
from .report import ReportGenerator

__all__ = [
    "Config",
    "TextChunk",
    "TextSplitter",
    "EmbeddingService",
    "VectorStore",
    "PlagiarismDetector",
    "PlagiarismReport",
    "Match",
    "ReportGenerator",
]