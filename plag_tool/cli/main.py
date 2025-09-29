"""Command-line interface for plag-tool."""

import sys
import logging
from pathlib import Path
import click

from ..core import (
    Config,
    PlagiarismDetector,
    ReportGenerator
)


# Configure logging
def setup_logging(verbose: bool):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="plag-tool")
def cli():
    """Plagiarism detection tool for Chinese and English text using vector embeddings."""
    pass


@cli.command()
@click.argument('source_file', type=click.Path(exists=True, path_type=Path))
@click.argument('target_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'html', 'text']), default='json', help='Output format')
@click.option('--threshold', '-t', type=float, default=0.85, help='Similarity threshold (0-1)')
@click.option('--chunk-size', '-c', type=int, default=500, help='Chunk size in characters')
@click.option('--overlap', type=int, default=100, help='Overlap size in characters')
@click.option('--sentence-boundaries', '-s', is_flag=True, default=True, help='Use sentence-aware chunking')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key (can be set via OPENAI_API_KEY env var)')
@click.option('--base-url', envvar='OPENAI_BASE_URL', help='OpenAI base URL (can be set via OPENAI_BASE_URL env var)')
@click.option('--model', envvar='OPENAI_MODEL', default='text-embedding-3-small', help='Embedding model to use')
def compare(
    source_file: Path,
    target_file: Path,
    output: Path,
    format: str,
    threshold: float,
    chunk_size: int,
    overlap: int,
    sentence_boundaries: bool,
    verbose: bool,
    api_key: str,
    base_url: str,
    model: str
):
    """
    Compare two files for plagiarism detection.

    SOURCE_FILE: Path to the source document
    TARGET_FILE: Path to the target document to check against
    """
    setup_logging(verbose)

    # Create configuration
    config = Config(
        similarity_threshold=threshold,
        chunk_size=chunk_size,
        overlap_size=overlap
    )

    # Override with CLI arguments if provided
    if api_key:
        config.openai_api_key = api_key
    if base_url:
        config.openai_base_url = base_url
    if model:
        config.openai_model = model

    # Check API key
    if not config.validate_api_key():
        click.echo("Error: OPENAI_API_KEY not set. Please provide it via --api-key or environment variable.", err=True)
        sys.exit(1)

    # Set default output path if not provided
    if not output:
        source_stem = source_file.stem
        output = Path(f"report_{source_stem}.{format}")

    click.echo(f"🔍 Comparing files for plagiarism...")
    click.echo(f"   Source: {source_file}")
    click.echo(f"   Target: {target_file}")
    click.echo(f"   Model: {config.openai_model}")
    click.echo(f"   Threshold: {threshold:.0%}")

    # Initialize detector
    detector = PlagiarismDetector(config)

    # Run detection
    with click.progressbar(length=100, label='Processing') as bar:
        bar.update(10)

        try:
            report = detector.compare_documents(
                str(source_file),
                str(target_file),
                use_sentence_boundaries=sentence_boundaries
            )
            bar.update(90)
        except Exception as e:
            click.echo(f"\n❌ Error during detection: {str(e)}", err=True)
            sys.exit(1)

    # Generate and save report
    generator = ReportGenerator()
    generator.save_report(report, str(output), format)

    # Display summary
    click.echo(f"\n✅ Analysis complete!")
    click.echo(f"   Plagiarism: {report.plagiarism_percentage:.1f}%")
    click.echo(f"   Matches found: {report.total_matches}")
    click.echo(f"   Report saved: {output}")

    # Show cost estimation if available
    if 'cost_estimation' in report.metadata:
        cost = report.metadata['cost_estimation']
        click.echo(f"   Estimated cost: ${cost.get('estimated_cost_usd', 0):.4f} USD")


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--chunk-size', '-c', type=int, default=500, help='Chunk size in characters')
@click.option('--overlap', type=int, default=100, help='Overlap size in characters')
@click.option('--sentence-boundaries', '-s', is_flag=True, default=True, help='Use sentence-aware chunking')
def analyze(file_path: Path, chunk_size: int, overlap: int, sentence_boundaries: bool):
    """
    Analyze a file to show chunking statistics without running detection.

    FILE_PATH: Path to the file to analyze
    """
    from ..core import TextChunker

    click.echo(f"📊 Analyzing file: {file_path}")

    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        click.echo(f"Error reading file: {str(e)}", err=True)
        sys.exit(1)

    # Create chunker
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    # Chunk text
    if sentence_boundaries:
        chunks = chunker.chunk_with_sentences(text, "analysis")
    else:
        chunks = chunker.chunk_text(text, "analysis")

    # Display statistics
    click.echo(f"\n📈 Chunking Statistics:")
    click.echo(f"   File length: {len(text):,} characters")
    click.echo(f"   Chunk size: {chunk_size} characters")
    click.echo(f"   Overlap: {overlap} characters")
    click.echo(f"   Total chunks: {len(chunks)}")

    if chunks:
        avg_chunk_size = sum(len(c.text) for c in chunks) / len(chunks)
        min_chunk_size = min(len(c.text) for c in chunks)
        max_chunk_size = max(len(c.text) for c in chunks)

        click.echo(f"   Average chunk size: {avg_chunk_size:.0f} characters")
        click.echo(f"   Min chunk size: {min_chunk_size} characters")
        click.echo(f"   Max chunk size: {max_chunk_size} characters")

        # Show first few chunks as preview
        click.echo(f"\n📝 Preview of first 3 chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk.text[:100]
            if len(chunk.text) > 100:
                preview += "..."
            click.echo(f"\n   Chunk {i} [{chunk.start_pos}:{chunk.end_pos}]:")
            click.echo(f"   {preview}")


@cli.command()
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--base-url', envvar='OPENAI_BASE_URL', help='OpenAI base URL')
@click.option('--model', envvar='OPENAI_MODEL', default='text-embedding-3-small', help='Embedding model')
def test_connection(api_key: str, base_url: str, model: str):
    """Test connection to the OpenAI API."""
    from openai import OpenAI

    config = Config()
    if api_key:
        config.openai_api_key = api_key
    if base_url:
        config.openai_base_url = base_url
    if model:
        config.openai_model = model

    if not config.validate_api_key():
        click.echo("❌ Error: OPENAI_API_KEY not set.", err=True)
        sys.exit(1)

    click.echo(f"🔌 Testing connection to API...")
    click.echo(f"   Endpoint: {config.openai_base_url}")
    click.echo(f"   Model: {config.openai_model}")

    try:
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

        # Test with a simple embedding
        response = client.embeddings.create(
            model=config.openai_model,
            input=["Test connection"]
        )

        click.echo(f"\n✅ Connection successful!")
        click.echo(f"   Embedding dimension: {len(response.data[0].embedding)}")

    except Exception as e:
        click.echo(f"\n❌ Connection failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('text1')
@click.argument('text2')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--base-url', envvar='OPENAI_BASE_URL', help='OpenAI base URL')
@click.option('--model', envvar='OPENAI_MODEL', default='text-embedding-3-small', help='Embedding model')
def quick_compare(text1: str, text2: str, api_key: str, base_url: str, model: str):
    """
    Quick comparison of two text strings.

    TEXT1: First text string
    TEXT2: Second text string
    """
    import numpy as np
    from ..core import EmbeddingService

    config = Config()
    if api_key:
        config.openai_api_key = api_key
    if base_url:
        config.openai_base_url = base_url
    if model:
        config.openai_model = model

    if not config.validate_api_key():
        click.echo("❌ Error: OPENAI_API_KEY not set.", err=True)
        sys.exit(1)

    click.echo("🔄 Generating embeddings...")

    try:
        embedder = EmbeddingService(config)

        # Generate embeddings
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)

        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        click.echo(f"\n📊 Results:")
        click.echo(f"   Text 1 length: {len(text1)} characters")
        click.echo(f"   Text 2 length: {len(text2)} characters")
        click.echo(f"   Similarity: {similarity:.2%}")

        if similarity >= 0.85:
            click.echo(f"   ⚠️  High similarity detected - possible plagiarism")
        elif similarity >= 0.50:
            click.echo(f"   ⚡ Moderate similarity - some related content")
        else:
            click.echo(f"   ✅ Low similarity - texts appear different")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()