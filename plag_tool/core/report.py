"""Report generation module for plagiarism detection results."""

import json
import html
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .detector import PlagiarismReport, Match


class ReportGenerator:
    """Generates various report formats for plagiarism detection results."""

    def __init__(self):
        """Initialize the report generator."""
        pass

    def generate_json(self, report: PlagiarismReport, indent: int = 2) -> str:
        """
        Generate JSON format report.

        Args:
            report: PlagiarismReport object
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(report.dict(), ensure_ascii=False, indent=indent)

    def generate_text(self, report: PlagiarismReport, max_matches: int = 10) -> str:
        """
        Generate plain text format report.

        Args:
            report: PlagiarismReport object
            max_matches: Maximum number of matches to display

        Returns:
            Plain text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PLAGIARISM DETECTION REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # File information
        lines.append("FILES ANALYZED:")
        lines.append(f"  Source: {report.source_file}")
        lines.append(f"          Length: {report.source_length:,} characters")
        lines.append(f"  Target: {report.target_file}")
        lines.append(f"          Length: {report.target_length:,} characters")
        lines.append("")

        # Detection settings
        lines.append("DETECTION SETTINGS:")
        lines.append(f"  Similarity Threshold: {report.detection_threshold:.2f}")
        lines.append(f"  Model: {report.metadata.get('embedding_model', 'N/A')}")
        lines.append(f"  Source Chunks: {report.metadata.get('source_chunks', 0)}")
        lines.append(f"  Target Chunks: {report.metadata.get('target_chunks', 0)}")
        lines.append("")

        # Results summary
        lines.append("RESULTS SUMMARY:")
        lines.append(f"  Total Matches Found: {report.total_matches}")
        lines.append(f"  Plagiarism Percentage: {report.plagiarism_percentage:.1f}%")
        lines.append("")

        # Cost estimation if available
        if 'cost_estimation' in report.metadata:
            cost = report.metadata['cost_estimation']
            lines.append("COST ESTIMATION:")
            lines.append(f"  Total Tokens: {cost.get('total_tokens', 0):,}")
            lines.append(f"  Estimated Cost: ${cost.get('estimated_cost_usd', 0):.4f} USD")
            lines.append("")

        # Detailed matches
        if report.matches:
            lines.append("TOP MATCHES:")
            lines.append("-" * 60)

            for i, match in enumerate(report.matches[:max_matches], 1):
                lines.append(f"\nMatch #{i} (Similarity: {match.similarity:.2%})")
                lines.append(f"Position: Source[{match.source_start}:{match.source_end}] → Target[{match.target_start}:{match.target_end}]")

                # Source text preview
                source_preview = match.source_text[:200]
                if len(match.source_text) > 200:
                    source_preview += "..."
                lines.append(f"Source: {source_preview}")

                # Target text preview
                target_preview = match.target_text[:200]
                if len(match.target_text) > 200:
                    target_preview += "..."
                lines.append(f"Target: {target_preview}")

                # Exact matches
                if match.exact_matches:
                    lines.append(f"Exact Phrases: {', '.join(match.exact_matches[:3])}")

                lines.append("-" * 60)

            if len(report.matches) > max_matches:
                lines.append(f"\n... and {len(report.matches) - max_matches} more matches")
        else:
            lines.append("No plagiarism detected.")

        return "\n".join(lines)

    def generate_html(self, report: PlagiarismReport) -> str:
        """
        Generate HTML format report with styling.

        Args:
            report: PlagiarismReport object

        Returns:
            HTML string
        """
        # Calculate severity level
        if report.plagiarism_percentage >= 50:
            severity = "high"
            color = "#dc3545"
        elif report.plagiarism_percentage >= 20:
            severity = "medium"
            color = "#ffc107"
        else:
            severity = "low"
            color = "#28a745"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Detection Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .plagiarism-meter {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .meter {{
            background: #e0e0e0;
            border-radius: 20px;
            overflow: hidden;
            height: 40px;
            position: relative;
        }}
        .meter-fill {{
            background: {color};
            height: 100%;
            border-radius: 20px;
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
        }}
        .matches {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .match {{
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .match-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        .match-similarity {{
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .match-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .text-block {{
            padding: 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .text-block h4 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
        }}
        .text-content {{
            font-family: "Courier New", monospace;
            font-size: 0.9em;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .exact-matches {{
            margin-top: 10px;
            padding: 10px;
            background: #fff3cd;
            border-radius: 4px;
            border: 1px solid #ffc107;
        }}
        .exact-matches h4 {{
            margin: 0 0 5px 0;
            color: #856404;
            font-size: 0.9em;
        }}
        .exact-match {{
            display: inline-block;
            background: #ffc107;
            color: #000;
            padding: 2px 6px;
            margin: 2px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .metadata {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 30px;
        }}
        .metadata h3 {{
            margin-top: 0;
        }}
        .metadata-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        .no-matches {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Plagiarism Detection Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Source: {html.escape(Path(report.source_file).name)}</p>
        <p>Target: {html.escape(Path(report.target_file).name)}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Matches</h3>
            <div class="value">{report.total_matches}</div>
        </div>
        <div class="summary-card">
            <h3>Source Length</h3>
            <div class="value">{report.source_length:,}</div>
        </div>
        <div class="summary-card">
            <h3>Target Length</h3>
            <div class="value">{report.target_length:,}</div>
        </div>
    </div>

    <div class="plagiarism-meter">
        <h3>Plagiarism Level: <span style="color: {color}">{severity.upper()}</span></h3>
        <div class="meter">
            <div class="meter-fill" style="width: {min(report.plagiarism_percentage, 100):.1f}%">
                {report.plagiarism_percentage:.1f}%
            </div>
        </div>
    </div>

    <div class="matches">
        <h2>Detected Matches</h2>
        {self._generate_matches_html(report.matches)}
    </div>

    <div class="metadata">
        <h3>Detection Metadata</h3>
        {self._generate_metadata_html(report.metadata)}
    </div>
</body>
</html>"""

        return html_content

    def _generate_matches_html(self, matches: list[Match]) -> str:
        """Generate HTML for matches section."""
        if not matches:
            return '<div class="no-matches">✅ No plagiarism detected</div>'

        html_parts = []
        for i, match in enumerate(matches[:20], 1):  # Show top 20 matches
            source_preview = html.escape(match.source_text[:300])
            if len(match.source_text) > 300:
                source_preview += "..."

            target_preview = html.escape(match.target_text[:300])
            if len(match.target_text) > 300:
                target_preview += "..."

            exact_matches_html = ""
            if match.exact_matches:
                exact_items = "".join(
                    f'<span class="exact-match">{html.escape(em)}</span>'
                    for em in match.exact_matches[:5]
                )
                exact_matches_html = f"""
                <div class="exact-matches">
                    <h4>Exact Matching Phrases:</h4>
                    {exact_items}
                </div>"""

            html_parts.append(f"""
            <div class="match">
                <div class="match-header">
                    <span>Match #{i}</span>
                    <span class="match-similarity">{match.similarity:.1%} similar</span>
                </div>
                <div class="match-content">
                    <div class="text-block">
                        <h4>Source [{match.source_start}:{match.source_end}]</h4>
                        <div class="text-content">{source_preview}</div>
                    </div>
                    <div class="text-block">
                        <h4>Target [{match.target_start}:{match.target_end}]</h4>
                        <div class="text-content">{target_preview}</div>
                    </div>
                </div>
                {exact_matches_html}
            </div>""")

        if len(matches) > 20:
            html_parts.append(f'<p style="text-align: center; color: #666;">... and {len(matches) - 20} more matches</p>')

        return "".join(html_parts)

    def _generate_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Generate HTML for metadata section."""
        items = []

        # Add key metadata items
        if 'embedding_model' in metadata:
            items.append(('Embedding Model', metadata['embedding_model']))
        if 'source_chunks' in metadata:
            items.append(('Source Chunks', f"{metadata['source_chunks']:,}"))
        if 'target_chunks' in metadata:
            items.append(('Target Chunks', f"{metadata['target_chunks']:,}"))

        # Add cost estimation if available
        if 'cost_estimation' in metadata:
            cost = metadata['cost_estimation']
            items.append(('Total Tokens', f"{cost.get('total_tokens', 0):,}"))
            items.append(('Estimated Cost', f"${cost.get('estimated_cost_usd', 0):.4f} USD"))

        # Add cache stats if available
        if 'cache_stats' in metadata:
            stats = metadata['cache_stats']
            items.append(('Cache Hit Rate', f"{stats.get('hit_rate', 0):.1%}"))

        html_parts = []
        for key, value in items:
            html_parts.append(f"""
            <div class="metadata-item">
                <span>{html.escape(key)}</span>
                <strong>{html.escape(str(value))}</strong>
            </div>""")

        return "".join(html_parts)

    def save_report(
        self,
        report: PlagiarismReport,
        output_path: str,
        format: str = "json"
    ):
        """
        Save report to file.

        Args:
            report: PlagiarismReport object
            output_path: Path to save the report
            format: Output format (json, html, text)
        """
        path = Path(output_path)

        if format == "json":
            content = self.generate_json(report)
        elif format == "html":
            content = self.generate_html(report)
        elif format == "text":
            content = self.generate_text(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)