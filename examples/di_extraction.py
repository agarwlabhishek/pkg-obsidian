"""
Simple example usage of the text extraction module.
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from da3_obsidian.di_text_extraction import DocumentExtractor, initialize_client

# Load environment variables (only in test/example scripts, not in package)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_document_text(file_path: str, force_model: str = None):
    """Extract text from a document with detailed structure."""
    try:
        extractor = DocumentExtractor()

        if force_model:
            result = extractor.extract_text(file_path, force_model=force_model)
        else:
            result = extractor.extract_text(file_path)

        logger.info(f"✓ Extracted text from {file_path}")
        logger.info(f"  Model: {result.model_used}")
        logger.info(f"  Pages: {result.page_count}")
        logger.info(f"  Text length: {len(result.text_content)} chars")
        logger.info(f"  Tables: {len(result.tables)}")
        logger.info(f"  Figures: {len(result.figures)}")
        logger.info(f"  Paragraphs: {len(result.paragraphs)}")
        logger.info(f"  Processing time: {result.processing_time:.2f}s")

        if result.key_value_pairs:
            logger.info(f"  Key-value pairs: {len(result.key_value_pairs)}")

        # Show page-level details
        for page in result.pages:
            logger.info(
                f"  Page {page.page_number}: {len(page.lines)} lines, {len(page.words)} words, {len(page.selection_marks)} selection marks"
            )

        # Show table details
        for i, table in enumerate(result.tables):
            logger.info(f"  Table {i+1}: {table.row_count}x{table.column_count} ({len(table.cells)} cells)")

        return result

    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        return None


def save_results(result, output_path):
    """Save extraction results with full structure to JSON file."""
    if not result:
        return

    try:
        # Create comprehensive output with structure preservation
        output_data = {
            "text_content": result.text_content,
            "model_used": result.model_used,
            "page_count": result.page_count,
            "processing_time": result.processing_time,
            "metadata": result.metadata,
            "pages": [
                {
                    "page_number": page.page_number,
                    "dimensions": {"width": page.width, "height": page.height, "unit": page.unit},
                    "angle": page.angle,
                    "line_count": len(page.lines),
                    "word_count": len(page.words),
                    "selection_marks_count": len(page.selection_marks),
                    "lines": page.lines[:5],  # First 5 lines as sample
                    "words": page.words[:10],  # First 10 words as sample
                    "selection_marks": page.selection_marks,
                }
                for page in result.pages
            ],
            "tables": [
                {
                    "table_index": i,
                    "dimensions": {"rows": table.row_count, "columns": table.column_count},
                    "cell_count": len(table.cells),
                    "cells": table.cells,
                    "bounding_regions": table.bounding_regions,
                }
                for i, table in enumerate(result.tables)
            ],
            "figures": result.figures,
            "paragraphs": [
                {
                    "content": para["content"][:100] + "..." if len(para["content"]) > 100 else para["content"],
                    "role": para.get("role"),
                    "bounding_regions": para["bounding_regions"],
                }
                for para in result.paragraphs
            ],
            "key_value_pairs": result.key_value_pairs,
            "features": {
                "file_name": result.features.file_name,
                "file_size": result.features.file_size,
                "has_tables": result.features.has_tables,
                "table_count": result.features.table_count,
                "processing_time": result.features.processing_time,
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")

    except Exception as e:
        logger.error(f"✗ Save failed: {e}")


def show_structure_preview(result):
    """Show a preview of the extracted document structure."""
    if not result:
        return

    logger.info("=== Document Structure Preview ===")

    # Text preview
    if result.text_content:
        preview = result.text_content[:200]
        if len(result.text_content) > 200:
            preview += "..."
        logger.info(f"Text preview: {preview}")

    # Key-value pairs preview
    if result.key_value_pairs:
        logger.info("Key-value pairs:")
        for kv in result.key_value_pairs[:3]:
            logger.info(f"  {kv['key']}: {kv['value']}")
        if len(result.key_value_pairs) > 3:
            logger.info(f"  ... and {len(result.key_value_pairs) - 3} more")

    # Table structure preview
    if result.tables:
        logger.info("Tables:")
        for i, table in enumerate(result.tables):
            logger.info(f"  Table {i+1}: {table.row_count} rows × {table.column_count} columns")
            if table.cells:
                sample_cells = table.cells[:3]
                for cell in sample_cells:
                    logger.info(f"    Cell [{cell['row_index']},{cell['column_index']}]: {cell['content'][:50]}...")

    # Figures preview
    if result.figures:
        logger.info(f"Figures: {len(result.figures)} detected")
        for i, figure in enumerate(result.figures):
            logger.info(f"  Figure {i+1}: ID={figure.get('id', 'N/A')}")

    # Pages preview
    logger.info("Pages:")
    for page in result.pages:
        logger.info(f"  Page {page.page_number}: {page.width}×{page.height} {page.unit}, {len(page.lines)} lines")


def main():
    """Run extraction example with enhanced structure."""
    # Test file path - replace with your actual document
    test_file = "sample_documents/sample.pdf"

    if not Path(test_file).exists():
        logger.warning(f"Test file '{test_file}' not found")
        logger.info("Please place a PDF file named 'sample_documents/sample.pdf' in the current directory")
        return

    try:
        # Initialize client
        initialize_client()
        logger.info("✓ Document Intelligence client initialized")

        # Extract with detailed structure
        result = extract_document_text(test_file)

        if result:
            # Show structure preview
            show_structure_preview(result)

            # Save comprehensive results
            save_results(result, "results/di_extraction_result.json")

            logger.info("\n✓ Extraction complete!")
            logger.info(f"  Total content: {len(result.text_content):,} characters")
            logger.info(f"  Document pages: {result.page_count}")
            logger.info(f"  Tables found: {len(result.tables)}")
            logger.info(f"  Figures found: {len(result.figures)}")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")

    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        logger.info("Check your AZURE_DI_ENDPOINT and AZURE_DI_KEY environment variables")


if __name__ == "__main__":
    main()
