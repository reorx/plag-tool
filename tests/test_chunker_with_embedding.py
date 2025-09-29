"""Integration test for chunker and embedding service."""

import os
import pytest
import numpy as np
from dotenv import load_dotenv

from plag_tool.core.config import Config
from plag_tool.core.chunker import TextChunker
from plag_tool.core.embeddings import EmbeddingService
from plag_tool.core.log import set_logger

# Load environment variables at module level
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_DEFAULT_EMBEDDING_MODEL"),
    reason="OPENAI_API_KEY not configured - skipping integration test"
)
def test_chunker_with_embedding_integration():
    """
    Integration test that combines chunker and embedding service.
    Uses a Chinese text of about 200 characters, chunks it, then embeds the chunks.
    """
    # Configure logging to DEBUG level to see chunk details
    set_logger(
        'plag_tool',
        level='DEBUG',
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        remove_handlers=True  # Clear any existing handlers first
    )

    # Setup config and services
    config = Config()

    # Skip if API key is not available
    if not config.validate_api_key():
        pytest.skip("API key not configured")

    # Chinese text about AI and technology (approximately 200 characters)
    chinese_text = """人工智能技术正在快速发展，深度学习算法在图像识别、自然语言处理等领域取得了重大突破。机器学习模型的性能不断提升，为各行各业带来了新的机遇和挑战。随着计算能力的增强和数据量的增长，人工智能在医疗诊断、自动驾驶、智能制造等方面展现出巨大潜力。未来人工智能将继续推动科技进步和社会发展。"""

    print(f"\n📝 Original text ({len(chinese_text)} characters):")
    print(f"'{chinese_text}'")

    # Initialize chunker with reasonable parameters
    chunker = TextChunker(chunk_size=50, overlap=10)
    print(f"\n🔄 Chunking with chunk_size=50, overlap=10")

    # Chunk the text
    chunks = chunker.chunk_text(chinese_text, "ai_article")

    print(f"\n📊 Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: pos {chunk.start_pos}-{chunk.end_pos}, "
              f"text: '{chunk.text}'")

    # Initialize embedding service
    embedding_service = EmbeddingService(config)

    # Embed the chunks with small batch settings to see batching in action
    print(f"\n🚀 Embedding chunks with batch limits (tokens=100, items=2)...")
    embeddings = embedding_service.embed_chunks(
        chunks,
        use_cache=False,  # Disable cache to see API calls
        batch_max_tokens=100,  # Small token limit to force multiple batches
        batch_max_items=2,     # Small item limit to force multiple batches
        max_retries=2
    )

    # Verify results
    assert len(embeddings) == len(chunks), f"Expected {len(chunks)} embeddings, got {len(embeddings)}"

    print(f"\n✅ Verification results:")
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        assert isinstance(embedding, np.ndarray), f"Embedding {i} should be numpy array"
        assert embedding.dtype in [np.float32, np.float64], f"Embedding {i} should be float type"
        assert len(embedding) > 0, f"Embedding {i} should have dimensions"
        assert np.all(np.isfinite(embedding)), f"Embedding {i} should have finite values"
        assert not np.allclose(embedding, 0), f"Embedding {i} should not be all zeros"

        print(f"  Chunk {i+1}: ✓ embedding shape {embedding.shape}, dtype {embedding.dtype}")

    print(f"\n🎉 Integration test completed successfully!")
    print(f"   - Processed {len(chinese_text)} character Chinese text")
    print(f"   - Created {len(chunks)} chunks")
    print(f"   - Generated {len(embeddings)} embeddings")
    print(f"   - Embedding dimension: {len(embeddings[0])}")
    print(f"   - Model used: {config.openai_model}")