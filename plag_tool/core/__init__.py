"""Core modules for plagiarism detection."""

from .config import Config
from .types import TextChunk, Match, PlagiarismReport
from .splitter import TextSplitter
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .detector import PlagiarismDetector
from .report import ReportGenerator

__all__ = [
    "Config",
    "TextChunk",
    "Match",
    "PlagiarismReport",
    "TextSplitter",
    "EmbeddingService",
    "VectorStore",
    "PlagiarismDetector",
    "ReportGenerator",
]