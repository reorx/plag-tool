"""Configuration module for plag-tool."""

import os
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config(BaseModel):
    """Configuration for the plagiarism detection system."""

    # OpenAI API settings
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="API key for OpenAI or compatible service"
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        description="Base URL for OpenAI-compatible API"
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_DEFAULT_EMBEDDING_MODEL") or
                               os.getenv("OPENAI_MODEL", "text-embedding-3-small"),
        description="Model name for embeddings"
    )

    # ChromaDB settings
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        description="Directory for ChromaDB persistence"
    )

    # Chunking settings
    chunk_size: int = Field(
        default=500,
        description="Size of text chunks in characters"
    )
    overlap_size: int = Field(
        default=100,
        description="Overlap between chunks in characters"
    )

    # Detection settings
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for plagiarism detection"
    )
    top_k: int = Field(
        default=10,
        description="Number of top similar chunks to retrieve"
    )

    # Processing settings
    batch_size: int = Field(
        default=100,
        description="Batch size for embedding generation"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for API calls"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )

    def validate_api_key(self) -> bool:
        """Check if API key is configured."""
        return bool(self.openai_api_key)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )