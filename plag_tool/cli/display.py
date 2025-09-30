"""Rich-based display module for plagiarism detection results."""

from typing import List
from rich.console import Console, RenderableType
from rich.text import Text
from rich.style import Style
from rich.table import Table

from ..core.types import Match, PlagiarismReport


def create_console() -> Console:
    """Create a rich Console instance."""
    return Console()


def comparison(renderable1: RenderableType, renderable2: RenderableType) -> Table:
    """
    Create a side-by-side comparison table with two columns.

    Args:
        renderable1: Content for the first column
        renderable2: Content for the second column

    Returns:
        A Table with two equal-width columns
    """
    table = Table(show_header=False, pad_edge=False, box=None, expand=True)
    table.add_column("1", ratio=1)
    table.add_column("2", ratio=1)
    table.add_row(renderable1, renderable2)
    return table


def highlight_exact_matches(text: str, exact_matches: List[str]) -> Text:
    """
    Highlight exact matching phrases in text.

    Args:
        text: The text to highlight
        exact_matches: List of exact matching phrases

    Returns:
        Rich Text object with highlights
    """
    # Sort matches by length (longest first) to handle overlapping matches
    sorted_matches = sorted(exact_matches, key=len, reverse=True)

    # Create rich Text object
    rich_text = Text(text)

    # Track which parts of the text have been highlighted
    highlighted_ranges = []

    for match_phrase in sorted_matches:
        if not match_phrase.strip():
            continue

        # Find all occurrences of this phrase
        start = 0
        while True:
            pos = text.find(match_phrase, start)
            if pos == -1:
                break

            end = pos + len(match_phrase)

            # Check if this range overlaps with already highlighted ranges
            overlaps = any(
                pos < r_end and end > r_start
                for r_start, r_end in highlighted_ranges
            )

            if not overlaps:
                highlighted_ranges.append((pos, end))

            start = end

    # Sort ranges by start position
    highlighted_ranges.sort()

    # Rebuild text with highlights
    if highlighted_ranges:
        rich_text = Text()
        last_pos = 0

        for start, end in highlighted_ranges:
            # Add unhighlighted text before this match
            if start > last_pos:
                rich_text.append(text[last_pos:start])

            # Add highlighted match
            rich_text.append(text[start:end], style="bold yellow")
            last_pos = end

        # Add remaining unhighlighted text
        if last_pos < len(text):
            rich_text.append(text[last_pos:])

    return rich_text


def truncate_with_highlights(text: str, exact_matches: List[str], max_length: int = 300) -> str:
    """
    Truncate text intelligently, preserving areas with highlights.

    Args:
        text: The text to truncate
        exact_matches: List of exact matching phrases
        max_length: Maximum length of output

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # If we have exact matches, try to center around the first match
    if exact_matches:
        first_match = exact_matches[0]
        match_pos = text.find(first_match)

        if match_pos != -1:
            # Calculate how much context to show before and after
            context_before = max(0, match_pos - max_length // 3)
            context_after = min(len(text), match_pos + len(first_match) + max_length // 3)

            # Extend to max_length if possible
            if context_after - context_before < max_length:
                if context_before > 0:
                    context_before = max(0, context_after - max_length)
                else:
                    context_after = min(len(text), context_before + max_length)

            prefix = "..." if context_before > 0 else ""
            suffix = "..." if context_after < len(text) else ""

            return prefix + text[context_before:context_after] + suffix

    # No matches or match not found, just truncate from start
    return text[:max_length] + "..."


def get_similarity_style(similarity: float) -> Style:
    """
    Get color style based on similarity score.

    Args:
        similarity: Similarity score (0-1)

    Returns:
        Rich Style object
    """
    if similarity >= 0.95:
        return Style(color="red", bold=True)
    elif similarity >= 0.85:
        return Style(color="yellow", bold=True)
    else:
        return Style(color="cyan", bold=True)


def display_match(console: Console, match: Match, index: int, max_text_length: int = 300):
    """
    Display a single match with rich formatting.

    Args:
        console: Rich Console instance
        match: Match object to display
        index: Match index number
        max_text_length: Maximum length for displayed text
    """
    # Create header with similarity score
    similarity_pct = f"{match.similarity:.1%}"
    similarity_style = get_similarity_style(match.similarity)

    header = Text()
    header.append(f"Match #{index}", style="bold cyan")
    header.append(" - ")
    header.append(similarity_pct, style=similarity_style)
    header.append(" similar")

    console.print(header)

    # Truncate texts if needed
    source_text = truncate_with_highlights(match.source_text, match.exact_matches, max_text_length)
    target_text = truncate_with_highlights(match.target_text, match.exact_matches, max_text_length)

    # Highlight texts
    source_highlighted = highlight_exact_matches(source_text, match.exact_matches)
    target_highlighted = highlight_exact_matches(target_text, match.exact_matches)

    # Create labels for source and target
    source_label = Text(f"Source [{match.source_start}:{match.source_end}]:", style="dim")
    target_label = Text(f"Target [{match.target_start}:{match.target_end}]:", style="dim")

    # Combine label and text for each column
    source_content = Text()
    source_content.append_text(source_label)
    source_content.append("\n")
    source_content.append_text(source_highlighted)

    target_content = Text()
    target_content.append_text(target_label)
    target_content.append("\n")
    target_content.append_text(target_highlighted)

    # Display side-by-side comparison
    console.print(comparison(source_content, target_content))

    # Display exact matches if any
    if match.exact_matches:
        console.print(f"Exact phrases: ", style="dim", end="")
        for i, phrase in enumerate(match.exact_matches[:3]):
            if i > 0:
                console.print(", ", end="")
            console.print(phrase, style="yellow", end="")
        if len(match.exact_matches) > 3:
            console.print(f" (+{len(match.exact_matches) - 3} more)", style="dim", end="")
        console.print()  # newline

    console.print()  # Empty line between matches


def display_summary(console: Console, report: PlagiarismReport):
    """
    Display summary statistics.

    Args:
        console: Rich Console instance
        report: PlagiarismReport object
    """
    console.print("Analysis complete!", style="bold green")
    console.print()

    # Plagiarism percentage with color coding
    pct = report.plagiarism_percentage
    if pct >= 50:
        pct_style = "bold red"
    elif pct >= 20:
        pct_style = "bold yellow"
    else:
        pct_style = "bold green"

    console.print(f"  Plagiarism: ", end="")
    console.print(f"{pct:.1f}%", style=pct_style)

    console.print(f"  Matches found: {report.total_matches}")

    # Cost estimation if available
    if 'cost_estimation' in report.metadata:
        cost = report.metadata['cost_estimation']
        console.print(f"  Estimated cost: ${cost.get('estimated_cost_usd', 0):.4f} USD", style="dim")

    console.print()


def display_report(report: PlagiarismReport, output_path: str, max_matches: int = 10):
    """
    Display plagiarism report with rich formatting.

    Args:
        report: PlagiarismReport object
        output_path: Path where report was saved
        max_matches: Maximum number of matches to display
    """
    console = create_console()

    # Display summary
    display_summary(console, report)

    # Display matches
    if report.matches:
        console.print(f"Top {min(len(report.matches), max_matches)} matches:", style="bold")
        console.print()

        for i, match in enumerate(report.matches[:max_matches], 1):
            display_match(console, match, i)

        if len(report.matches) > max_matches:
            console.print(f"... and {len(report.matches) - max_matches} more matches (see full report)", style="dim")
            console.print()
    else:
        console.print("No plagiarism detected.", style="bold green")
        console.print()

    # Display report location
    console.print(f"Full report saved to: {output_path}", style="cyan")
