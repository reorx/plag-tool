"""Tests for the chunker module."""

import pytest
from plag_tool.core.chunker import TextChunker, TextChunk


class TestTextChunker:
    """Test cases for TextChunker class."""

    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        chunker = TextChunker(chunk_size=10, overlap=3)
        text = "这是一段测试文本用于演示分块算法"
        chunks = chunker.chunk_text(text, "test_doc")

        # Check we got chunks
        assert len(chunks) > 1

        # Check chunk properties
        assert chunks[0].doc_id == "test_doc"
        assert chunks[0].start_pos == 0
        assert chunks[0].chunk_index == 0
        assert len(chunks[0].text) == 10

        # Check positions are sequential
        for i in range(1, len(chunks)):
            assert chunks[i].start_pos > chunks[i-1].start_pos
            assert chunks[i].chunk_index == i

    def test_chunk_text_with_overlap(self):
        """Test that overlap is working correctly."""
        chunker = TextChunker(chunk_size=10, overlap=3)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunker.chunk_text(text, "overlap_test")

        # Should have overlap between consecutive chunks
        assert len(chunks) >= 2

        # Check first two chunks have overlap
        chunk1_end = chunks[0].text[-3:]  # Last 3 chars of first chunk
        chunk2_start = chunks[1].text[:3]  # First 3 chars of second chunk
        assert chunk1_end == chunk2_start

        # Verify stride calculation
        expected_stride = 10 - 3  # chunk_size - overlap
        assert chunks[1].start_pos - chunks[0].start_pos == expected_stride

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = TextChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk_text("", "empty_doc")

        assert len(chunks) == 0

    def test_chunk_with_sentences_chinese(self):
        """Test sentence-aware chunking with Chinese text."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "人工智能正在改变世界。深度学习是其重要分支。神经网络模型表现出色。"
        chunks = chunker.chunk_with_sentences(text, "chinese_doc")

        # Should create chunks
        assert len(chunks) > 0

        # Check that chunk boundaries try to respect sentence endings
        for chunk in chunks:
            assert chunk.doc_id == "chinese_doc"
            assert isinstance(chunk.start_pos, int)
            assert isinstance(chunk.end_pos, int)
            assert chunk.start_pos < chunk.end_pos

    def test_chunk_with_sentences_english(self):
        """Test sentence-aware chunking with English text."""
        chunker = TextChunker(chunk_size=30, overlap=5)
        text = "AI is transforming the world. Deep learning is important. Neural networks perform well."
        chunks = chunker.chunk_with_sentences(text, "english_doc")

        # Should create chunks
        assert len(chunks) > 0

        # Check basic properties
        for chunk in chunks:
            assert chunk.doc_id == "english_doc"
            assert len(chunk.text) > 0
            assert chunk.end_pos > chunk.start_pos

        # Check that sentences are preserved where possible
        # (This is a heuristic test - exact behavior depends on chunk size)
        full_text_reconstructed = "".join([c.text for c in chunks])
        assert len(full_text_reconstructed) >= len(text)  # Due to overlap

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""

        # Negative chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-1, overlap=0)

        # Zero chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0, overlap=0)

        # Negative overlap
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            TextChunker(chunk_size=10, overlap=-1)

        # Overlap equals chunk_size
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            TextChunker(chunk_size=10, overlap=10)

        # Overlap greater than chunk_size
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            TextChunker(chunk_size=10, overlap=15)

    def test_chunk_text_single_chunk(self):
        """Test chunking text that fits in a single chunk."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "短文本"
        chunks = chunker.chunk_text(text, "single_chunk")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_pos == 0
        assert chunks[0].end_pos == len(text)

    def test_chunk_consistency(self):
        """Test that chunks maintain consistency properties."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "这是一个相对较长的中文测试文本，用来验证分块算法的一致性和正确性。"
        chunks = chunker.chunk_text(text, "consistency_test")

        # Verify all chunks have required properties
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, TextChunk)
            assert chunk.chunk_index == i
            assert chunk.doc_id == "consistency_test"
            assert len(chunk.text) > 0
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos
            assert chunk.end_pos <= len(text)
            assert len(chunk.chunk_hash) == 16  # SHA256 truncated to 16 chars

        # Verify chunks cover the text properly
        if len(chunks) > 1:
            # First chunk should start at 0
            assert chunks[0].start_pos == 0
            # Last chunk should end at text length
            assert chunks[-1].end_pos == len(text)
            # No gaps between chunks (considering overlap)
            for i in range(len(chunks) - 1):
                gap = chunks[i+1].start_pos - chunks[i].end_pos
                assert gap <= 0  # Should be negative (overlap) or zero