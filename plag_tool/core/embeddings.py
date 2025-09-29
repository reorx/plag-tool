"""Embedding service for converting text to vector representations."""

import time
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .chunker import TextChunk
from .log import base_logger

logger = base_logger.getChild('embeddings')


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        """Initialize the cache with a maximum size."""
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get an embedding from the cache."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: np.ndarray):
        """Put an embedding into the cache."""
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class EmbeddingService:
    """Service for generating embeddings using OpenAI-compatible API."""

    def __init__(self, config: Config):
        """
        Initialize the embedding service.

        Args:
            config: Configuration object with API settings
        """
        self.config = config

        # Log the initialization parameters
        logger.info("Initializing OpenAI client with:")
        logger.info(f"  Base URL: {config.openai_base_url}")
        api_key_display = ('***' + config.openai_api_key[-4:]
                          if len(config.openai_api_key) > 4
                          else '***')
        logger.info(f"  API Key: {api_key_display}")
        logger.info(f"  Model: {config.openai_model}")

        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        self.model = config.openai_model
        self.cache = EmbeddingCache()
        self.batch_size = config.batch_size
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay

        # Initialize tiktoken encoder for accurate token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            logger.warning(f"Unknown model {self.model}, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use caching

        Returns:
            Numpy array of the embedding
        """
        if use_cache:
            # Check cache first
            cache_key = f"{self.model}:{text[:100]}"  # Use first 100 chars as key
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Generate embedding
        embedding = self._call_api([text])[0]

        if use_cache:
            self.cache.put(cache_key, embedding)

        return embedding

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken for accurate counting.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def embed_chunks(self, chunks: List[TextChunk], use_cache: bool = True,
                    batch_max_tokens: Optional[int] = None,
                    batch_max_items: Optional[int] = None,
                    max_retries: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple text chunks with optimized batching.

        Args:
            chunks: List of TextChunk objects
            use_cache: Whether to use caching
            batch_max_tokens: Maximum tokens per batch (overrides config)
            batch_max_items: Maximum items per batch (overrides config)
            max_retries: Maximum retries for batch operations (overrides config)

        Returns:
            List of numpy arrays containing embeddings
        """
        # Use config defaults if parameters not provided
        effective_max_tokens = batch_max_tokens or self.config.embedding_batch_max_tokens
        effective_max_items = batch_max_items or self.config.embedding_batch_max_items
        effective_max_retries = max_retries or self.max_retries

        # Apply 10% safety margin to token limit
        safe_max_tokens = int(effective_max_tokens * 0.9)

        embeddings = []
        texts_to_embed = []
        cache_indices = []

        # Check cache for each chunk
        for i, chunk in enumerate(chunks):
            if use_cache:
                cache_key = f"{self.model}:{chunk.chunk_hash}"
                cached = self.cache.get(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    continue

            texts_to_embed.append(chunk.text)
            cache_indices.append(i)
            embeddings.append(None)  # Placeholder

        # Process uncached texts with optimized batching
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} chunks...")

            # Create retry-wrapped version of _batch_embed
            retried_batch_embed = retry(
                stop=stop_after_attempt(effective_max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10)
            )(self._batch_embed)

            # Split texts into optimized batches
            batches = self._create_token_limited_batches(
                texts_to_embed, safe_max_tokens, effective_max_items
            )

            # Log batch information
            logger.info(f"Splitting {len(texts_to_embed)} chunks into {len(batches)} batches "
                       f"(max_tokens: {safe_max_tokens}, max_items: {effective_max_items})")
            for i, batch in enumerate(batches):
                batch_tokens = sum(self.count_tokens(text) for text in batch)
                logger.info(f"  Batch {i+1}: {len(batch)} items, ~{batch_tokens} tokens")

            # Process each batch with retry logic
            all_new_embeddings = []
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} items)...")
                batch_embeddings = retried_batch_embed(batch)
                all_new_embeddings.extend(batch_embeddings)

            # Fill in the embeddings and update cache
            for idx, embedding in zip(cache_indices, all_new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    cache_key = f"{self.model}:{chunks[idx].chunk_hash}"
                    self.cache.put(cache_key, embedding)

        # Validate result length
        if len(embeddings) != len(chunks):
            raise RuntimeError(f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}")

        return embeddings

    def _create_token_limited_batches(self, texts: List[str], max_tokens: int, max_items: int) -> List[List[str]]:
        """
        Create batches of texts respecting both token and item limits.

        Args:
            texts: List of texts to batch
            max_tokens: Maximum tokens per batch
            max_items: Maximum items per batch

        Returns:
            List of batches (each batch is a list of texts)
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self.count_tokens(text)

            # Check if adding this text would exceed limits
            would_exceed_tokens = current_tokens + text_tokens > max_tokens
            would_exceed_items = len(current_batch) >= max_items

            # Start new batch if either limit would be exceeded
            if current_batch and (would_exceed_tokens or would_exceed_items):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            # Add text to current batch
            current_batch.append(text)
            current_tokens += text_tokens

        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _call_api(self, texts: List[str]) -> List[np.ndarray]:
        """
        Call the OpenAI API with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings as numpy arrays
        """
        # Debug logging: show what chunks are being sent to OpenAI
        total_tokens = sum(self.count_tokens(text) for text in texts)
        logger.debug(f"Sending {len(texts)} texts to OpenAI API (total ~{total_tokens} tokens):")
        for i, text in enumerate(texts):
            # Show first 100 characters of each text for debugging
            preview = text[:100] + "..." if len(text) > 100 else text
            tokens = self.count_tokens(text)
            logger.debug(f"  Text {i+1}: {tokens} tokens - '{preview}'")

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )

                embeddings = []
                for item in response.data:
                    embedding = np.array(item.embedding, dtype=np.float32)
                    embeddings.append(embedding)

                return embeddings

            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        raise RuntimeError(f"Failed to generate embeddings after {self.max_retries} attempts: {str(last_error)}")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        This is a rough estimate; actual token count may vary.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        # Rough estimation: ~1 token per 4 characters for English
        # ~1 token per 2 characters for Chinese
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars

        return (chinese_chars // 2) + (other_chars // 4)

    def estimate_cost(self, texts: List[str]) -> Dict[str, float]:
        """
        Estimate the cost of embedding texts.

        Args:
            texts: List of texts to embed

        Returns:
            Dictionary with cost estimates
        """
        total_tokens = sum(self.estimate_tokens(text) for text in texts)

        # Pricing per 1M tokens (adjust based on model)
        pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }

        price_per_million = pricing.get(self.model, 0.02)
        estimated_cost = (total_tokens / 1_000_000) * price_per_million

        return {
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model,
            "price_per_million_tokens": price_per_million
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.cache.get_stats()