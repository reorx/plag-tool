"""Core plagiarism detection logic."""

import chardet
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field

from .config import Config
from .types import TextChunk, Match, PlagiarismReport
from .splitter import TextSplitter
from .embeddings import EmbeddingService
from .vector_store import VectorStore, SimilarityMatch
from .log import base_logger

logger = base_logger.getChild('detector')


class PlagiarismDetector:
    """Main plagiarism detection engine."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the plagiarism detector.

        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or Config()
        self.splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap_size
        )
        self.embedder = EmbeddingService(self.config)
        self.vector_store = VectorStore(self.config.chroma_persist_dir)

    def read_file(self, file_path: str) -> str:
        """
        Read a file with automatic encoding detection.

        Args:
            file_path: Path to the file

        Returns:
            File contents as string
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            confidence = result['confidence'] or 0

        logger.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")

        # Try detected encoding first, fallback to common Chinese encodings
        encodings_to_try = [encoding, 'utf-8', 'gb2312', 'gbk', 'gb18030', 'big5']

        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read()
                    logger.info(f"Successfully read {file_path} with encoding: {enc}")
                    return text
            except (UnicodeDecodeError, LookupError):
                continue

        raise ValueError(f"Could not decode file {file_path} with any known encoding")

    def compare_documents(
        self,
        source_file: str,
        target_file: str,
        use_sentence_boundaries: bool = True,
        force_embed: bool = False
    ) -> PlagiarismReport:
        """
        Compare two documents for plagiarism.

        Args:
            source_file: Path to source document
            target_file: Path to target document
            use_sentence_boundaries: Whether to use sentence-aware chunking
            force_embed: Force re-embedding even if embeddings exist

        Returns:
            PlagiarismReport with detection results
        """
        logger.info(f"Comparing {source_file} with {target_file}")

        # Read files
        source_text = self.read_file(source_file)
        target_text = self.read_file(target_file)

        # Create stable collection name based on chunking parameters
        import hashlib
        params_str = f"{self.config.chunk_size}_{self.config.overlap_size}_{use_sentence_boundaries}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        collection_name = f"plag_{params_hash}"

        # Create or get collection (non-destructive by default)
        self.vector_store.create_collection(collection_name, reset=force_embed)

        # Check if documents already have embeddings
        source_exists = self.vector_store.has_document("source")
        target_exists = self.vector_store.has_document("target")

        # Process source document
        if not force_embed and source_exists:
            logger.info("Using existing source embeddings from database")
            source_chunks = self.vector_store.get_document_text_chunks("source")
        else:
            logger.info(f"Splitting and embedding source document (force_embed={force_embed})")
            if use_sentence_boundaries:
                source_chunks = self.splitter.chunk_with_sentences(source_text, "source")
            else:
                source_chunks = self.splitter.chunk_text(source_text, "source")

        # Process target document
        if not force_embed and target_exists:
            logger.info("Using existing target embeddings from database")
            target_chunks = self.vector_store.get_document_text_chunks("target")
        else:
            logger.info(f"Splitting and embedding target document (force_embed={force_embed})")
            if use_sentence_boundaries:
                target_chunks = self.splitter.chunk_with_sentences(target_text, "target")
            else:
                target_chunks = self.splitter.chunk_text(target_text, "target")

        logger.info(f"Using {len(source_chunks)} source chunks and {len(target_chunks)} target chunks")

        # Generate and store embeddings for target if not using cached
        if force_embed or not target_exists:
            logger.info("Generating target embeddings...")
            target_embeddings = self.embedder.embed_chunks(target_chunks)
            # Delete existing target chunks if force_embed
            if force_embed and target_exists:
                self._delete_document_chunks("target")
            self.vector_store.add_documents(target_chunks, target_embeddings)

        # Generate embeddings for source and find matches
        if force_embed or not source_exists:
            logger.info("Generating source embeddings and finding matches...")
            source_embeddings = self.embedder.embed_chunks(source_chunks)
            # Delete existing source chunks if force_embed
            if force_embed and source_exists:
                self._delete_document_chunks("source")
            # Store source embeddings for future use
            self.vector_store.add_documents(source_chunks, source_embeddings)
        else:
            logger.info("Generating source embeddings for matching...")
            source_embeddings = self.embedder.embed_chunks(source_chunks)

        matches = self.find_matches(source_chunks, source_embeddings)

        # Calculate plagiarism statistics
        matched_chars = self._calculate_matched_characters(matches)
        total_chars = len(source_text)
        plagiarism_pct = (matched_chars / total_chars * 100) if total_chars > 0 else 0

        # Get cost estimation
        cost_info = self.embedder.estimate_cost([c.text for c in source_chunks + target_chunks])

        # Create report
        report = PlagiarismReport(
            source_file=source_file,
            target_file=target_file,
            total_matches=len(matches),
            plagiarism_percentage=plagiarism_pct,
            matches=matches,
            source_length=len(source_text),
            target_length=len(target_text),
            detection_threshold=self.config.similarity_threshold,
            metadata={
                "source_chunks": len(source_chunks),
                "target_chunks": len(target_chunks),
                "collection_name": collection_name,
                "embedding_model": self.config.openai_model,
                "cost_estimation": cost_info,
                "cache_stats": self.embedder.get_cache_stats()
            }
        )

        logger.info(f"Detection complete: {report.total_matches} matches found, {report.plagiarism_percentage:.1f}% plagiarism")

        return report

    def find_matches(
        self,
        source_chunks: List[TextChunk],
        source_embeddings: List
    ) -> List[Match]:
        """
        Find plagiarized segments between source and target.

        Args:
            source_chunks: List of source text chunks
            source_embeddings: List of source embeddings

        Returns:
            List of Match objects
        """
        all_matches = []

        for chunk, embedding in zip(source_chunks, source_embeddings):
            # Search for similar chunks in vector store
            similar_chunks = self.vector_store.search_similar(
                embedding,
                threshold=self.config.similarity_threshold,
                top_k=self.config.top_k,
                filter_doc_id="source"  # Exclude source document itself
            )

            # Create Match objects for each similar chunk
            for similar in similar_chunks:
                match = Match(
                    source_text=chunk.text,
                    source_start=chunk.start_pos,
                    source_end=chunk.end_pos,
                    target_text=similar.text,
                    target_start=similar.start_pos,
                    target_end=similar.end_pos,
                    similarity=similar.similarity
                )

                # Find exact matching phrases
                exact_matches = self.find_exact_matches(chunk.text, similar.text)
                match.exact_matches = exact_matches

                all_matches.append(match)

        # Merge overlapping matches
        merged_matches = self.merge_overlapping_matches(all_matches)

        return merged_matches

    def find_exact_matches(
        self,
        source: str,
        target: str,
        min_length: int = 10,
        min_words: int = 3
    ) -> List[str]:
        """
        Find exact matching substrings between two texts.

        Args:
            source: Source text
            target: Target text
            min_length: Minimum length of match in characters
            min_words: Minimum number of words for a match

        Returns:
            List of exact matching phrases
        """
        matches = []

        # Simple word-based matching
        source_words = source.split()
        target_words = target.split()

        # Find all common substrings
        for i in range(len(source_words)):
            for j in range(len(target_words)):
                # Check if words match
                if source_words[i] == target_words[j]:
                    # Extend the match
                    k = 0
                    while (i + k < len(source_words) and
                           j + k < len(target_words) and
                           source_words[i + k] == target_words[j + k]):
                        k += 1

                    # Check if match meets criteria
                    if k >= min_words:
                        match_text = ' '.join(source_words[i:i+k])
                        if len(match_text) >= min_length and match_text not in matches:
                            matches.append(match_text)

        # For Chinese text without spaces, use character-based matching
        if not matches and not ' ' in source:
            matches.extend(self._find_chinese_exact_matches(source, target, min_length))

        return sorted(matches, key=len, reverse=True)[:10]  # Return top 10 longest matches

    def _find_chinese_exact_matches(
        self,
        source: str,
        target: str,
        min_length: int = 10
    ) -> List[str]:
        """
        Find exact matches in Chinese text (character-based).

        Args:
            source: Source text
            target: Target text
            min_length: Minimum length of match

        Returns:
            List of matching substrings
        """
        matches = []
        source_len = len(source)
        target_len = len(target)

        # Dynamic programming approach for longest common substrings
        dp = [[0] * (target_len + 1) for _ in range(source_len + 1)]

        for i in range(1, source_len + 1):
            for j in range(1, target_len + 1):
                if source[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1

                    # Check if we have a long enough match
                    if dp[i][j] >= min_length:
                        match = source[i-dp[i][j]:i]
                        if match not in matches:
                            matches.append(match)

        return matches

    def merge_overlapping_matches(
        self,
        matches: List[Match],
        gap_tolerance: int = 50
    ) -> List[Match]:
        """
        Merge overlapping or adjacent matches.

        Args:
            matches: List of matches to merge
            gap_tolerance: Maximum gap between matches to merge

        Returns:
            List of merged matches
        """
        if not matches:
            return []

        # Sort by source position
        sorted_matches = sorted(matches, key=lambda m: m.source_start)
        merged = []
        current = sorted_matches[0]

        for next_match in sorted_matches[1:]:
            # Check if matches are close enough to merge
            if (next_match.source_start <= current.source_end + gap_tolerance and
                next_match.similarity >= self.config.similarity_threshold):

                # Merge matches
                current = Match(
                    source_text=self._merge_texts(current.source_text, next_match.source_text),
                    source_start=current.source_start,
                    source_end=max(current.source_end, next_match.source_end),
                    target_text=self._merge_texts(current.target_text, next_match.target_text),
                    target_start=min(current.target_start, next_match.target_start),
                    target_end=max(current.target_end, next_match.target_end),
                    similarity=max(current.similarity, next_match.similarity),
                    exact_matches=list(set(current.exact_matches + next_match.exact_matches))
                )
            else:
                merged.append(current)
                current = next_match

        merged.append(current)
        return merged

    def _merge_texts(self, text1: str, text2: str) -> str:
        """
        Merge two text segments, avoiding duplication.

        Args:
            text1: First text segment
            text2: Second text segment

        Returns:
            Merged text
        """
        # Simple approach: if text2 starts with end of text1, merge them
        overlap_len = min(len(text1), len(text2), 100)

        for i in range(overlap_len, 0, -1):
            if text1[-i:] == text2[:i]:
                return text1 + text2[i:]

        # No overlap found, concatenate with separator
        return text1 + " ... " + text2

    def _delete_document_chunks(self, doc_id: str):
        """
        Delete all chunks for a specific document from the vector store.

        Args:
            doc_id: Document identifier
        """
        if not self.vector_store.collection:
            return

        results = self.vector_store.collection.get(
            where={"doc_id": doc_id}
        )

        if results and results['ids']:
            self.vector_store.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} existing chunks for doc_id={doc_id}")

    def _calculate_matched_characters(self, matches: List[Match]) -> int:
        """
        Calculate total number of matched characters, avoiding double counting.

        Args:
            matches: List of matches

        Returns:
            Total number of unique matched characters
        """
        if not matches:
            return 0

        # Create intervals for matched regions
        intervals = [(m.source_start, m.source_end) for m in matches]

        # Merge overlapping intervals
        intervals.sort()
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                # Overlapping, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                # Non-overlapping, add new interval
                merged.append((start, end))

        # Calculate total characters
        total = sum(end - start for start, end in merged)

        return total
