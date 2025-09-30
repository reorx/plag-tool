"""Integration tests for the embeddings module."""

import os
import pytest
import numpy as np
from dotenv import load_dotenv

from plag_tool.core.config import Config
from plag_tool.core.embeddings import EmbeddingService
from plag_tool.core.splitter import TextChunk

# Load environment variables at module level
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_DEFAULT_EMBEDDING_MODEL"),
    reason="OPENAI_API_KEY not configured - skipping integration test"
)
def test_embed_chinese_text():
    """Test embedding Chinese text with actual API call."""
    # Setup
    config = Config()

    # Skip if API key is not available
    if not config.validate_api_key():
        pytest.skip("API key not configured")

    service = EmbeddingService(config)

    # Test data - Chinese paragraph about AI
    chinese_text = "人工智能技术正在快速发展，深度学习算法在图像识别、自然语言处理等领域取得了重大突破。机器学习模型的性能不断提升，为各行各业带来了新的机遇和挑战。"

    # Execute
    embedding = service.embed_text(chinese_text)

    # Verify the result is a numpy array of floats
    assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
    assert embedding.dtype in [np.float32, np.float64], f"Embedding should be float type, got {embedding.dtype}"
    assert len(embedding) > 0, "Embedding should have dimensions"
    assert embedding.ndim == 1, "Embedding should be 1-dimensional"

    # Check that all values are finite (no NaN or infinity)
    assert np.all(np.isfinite(embedding)), "All embedding values should be finite"

    # Check that it's not all zeros (which would indicate an error)
    assert not np.allclose(embedding, 0), "Embedding should not be all zeros"

    # For text-embedding-3-small, dimension should be 1536
    # For text-embedding-3-large, dimension should be 3072
    # We'll just check it's a reasonable size
    assert 100 < len(embedding) < 5000, f"Embedding dimension {len(embedding)} seems unreasonable"

    print(f"✅ Successfully generated embedding for Chinese text")
    print(f"   Text length: {len(chinese_text)} characters")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   Embedding type: {embedding.dtype}")
    print(f"   Model used: {config.openai_model}")


def test_token_counting():
    """Test token counting functionality."""
    config = Config()
    service = EmbeddingService(config)

    # Test English text
    english_text = "Hello world, this is a test."
    english_tokens = service.count_tokens(english_text)
    assert english_tokens > 0, "Should count tokens for English text"

    # Test Chinese text
    chinese_text = "你好世界，这是一个测试。"
    chinese_tokens = service.count_tokens(chinese_text)
    assert chinese_tokens > 0, "Should count tokens for Chinese text"

    # Test empty text
    empty_tokens = service.count_tokens("")
    assert empty_tokens == 0, "Empty text should have 0 tokens"

    print(f"✅ Token counting test passed")
    print(f"   English text: '{english_text}' -> {english_tokens} tokens")
    print(f"   Chinese text: '{chinese_text}' -> {chinese_tokens} tokens")


def test_batch_splitting():
    """Test batch splitting with token and item limits."""
    config = Config()
    service = EmbeddingService(config)

    # Create test texts with varying sizes
    texts = [
        "Short text",
        "Medium length text with more words to test batching",
        "Very long text that contains many words and should contribute significantly to token count for testing batch splitting functionality",
        "Another medium text",
        "Short",
        "Final longer text to ensure we test multiple scenarios for batch creation"
    ]

    # Test with small limits to force multiple batches
    batches = service._create_token_limited_batches(texts, max_tokens=50, max_items=3)

    assert len(batches) > 1, "Should create multiple batches with small limits"

    # Verify each batch respects limits
    for i, batch in enumerate(batches):
        assert len(batch) <= 3, f"Batch {i} exceeds item limit"
        batch_tokens = sum(service.count_tokens(text) for text in batch)
        # Note: we allow the first item even if it exceeds token limit
        if len(batch) > 1:
            assert batch_tokens <= 50, f"Batch {i} exceeds token limit: {batch_tokens} tokens"

    # Verify all texts are included
    all_texts_in_batches = []
    for batch in batches:
        all_texts_in_batches.extend(batch)
    assert len(all_texts_in_batches) == len(texts), "All texts should be included in batches"
    assert set(all_texts_in_batches) == set(texts), "All original texts should be preserved"

    print(f"✅ Batch splitting test passed")
    print(f"   Created {len(batches)} batches from {len(texts)} texts")
    for i, batch in enumerate(batches):
        batch_tokens = sum(service.count_tokens(text) for text in batch)
        print(f"   Batch {i+1}: {len(batch)} items, {batch_tokens} tokens")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_DEFAULT_EMBEDDING_MODEL"),
    reason="OPENAI_API_KEY not configured - skipping integration test"
)
def test_embed_chunks_with_batching():
    """Test embedding chunks with optimized batching."""
    config = Config()

    # Skip if API key is not available
    if not config.validate_api_key():
        pytest.skip("API key not configured")

    service = EmbeddingService(config)

    # Create test chunks
    texts = [
        "人工智能技术正在快速发展。",
        "深度学习算法在图像识别领域取得突破。",
        "机器学习模型性能不断提升。",
        "自然语言处理技术日益成熟。",
        "大数据分析为企业提供洞察。"
    ]

    chunks = []
    for i, text in enumerate(texts):
        chunk = TextChunk(
            text=text,
            start_pos=i * 100,
            end_pos=(i + 1) * 100,
            doc_id="test_doc",
            chunk_index=i
        )
        chunks.append(chunk)

    # Test with small batch limits to force batching
    embeddings = service.embed_chunks(
        chunks,
        use_cache=False,
        batch_max_tokens=100,  # Small token limit
        batch_max_items=2,     # Small item limit
        max_retries=2
    )

    # Verify results
    assert len(embeddings) == len(chunks), "Should return embedding for each chunk"

    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, np.ndarray), f"Embedding {i} should be numpy array"
        assert embedding.dtype in [np.float32, np.float64], f"Embedding {i} should be float type"
        assert len(embedding) > 0, f"Embedding {i} should have dimensions"
        assert np.all(np.isfinite(embedding)), f"Embedding {i} should have finite values"
        assert not np.allclose(embedding, 0), f"Embedding {i} should not be all zeros"

    print(f"✅ Embed chunks with batching test passed")
    print(f"   Successfully embedded {len(chunks)} chunks")
    print(f"   Embedding dimensions: {len(embeddings[0])}")


def test_config_environment_variables():
    """Test that config properly reads environment variables."""
    # Test with default values
    config = Config()

    assert hasattr(config, 'embedding_batch_max_tokens'), "Config should have embedding_batch_max_tokens"
    assert hasattr(config, 'embedding_batch_max_items'), "Config should have embedding_batch_max_items"
    assert config.embedding_batch_max_tokens > 0, "Token limit should be positive"
    assert config.embedding_batch_max_items > 0, "Item limit should be positive"

    print(f"✅ Config environment variables test passed")
    print(f"   Max tokens per batch: {config.embedding_batch_max_tokens}")
    print(f"   Max items per batch: {config.embedding_batch_max_items}")