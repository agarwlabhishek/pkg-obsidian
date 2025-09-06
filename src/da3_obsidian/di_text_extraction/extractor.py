"""
Main document extraction interface using Azure Document Intelligence.

Provides comprehensive document analysis and text extraction capabilities with
automatic model selection and multi-format support.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from azure.core.exceptions import HttpResponseError

from .engines import get_document_client, initialize_client, is_client_initialized
from .models import (
    DOCUMENT_TYPE_INDICATORS,
    MODEL_DETECTION_CONFIG,
    MODEL_TYPES,
    SUPPORTED_FILE_EXTENSIONS,
    get_content_type,
    get_model_id,
    validate_file_extension,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    """Represents a single page from the document."""

    page_number: int
    width: float
    height: float
    unit: str
    angle: float
    lines: List[Dict[str, Any]]
    words: List[Dict[str, Any]]
    selection_marks: List[Dict[str, Any]]


@dataclass
class DocumentTable:
    """Represents a table extracted from the document."""

    row_count: int
    column_count: int
    cells: List[Dict[str, Any]]
    bounding_regions: List[Dict[str, Any]]
    spans: List[Dict[str, Any]]


@dataclass
class DocumentFeatures:
    """Represents extracted features from a document for model selection."""

    file_name: str
    file_size: int
    file_extension: str
    text_content: str
    page_count: int
    key_value_pairs: List[Dict[str, str]]
    key_phrases: List[str]
    first_page_text: str
    has_tables: bool
    table_count: int
    processing_time: float
    error: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of document text extraction operation."""

    text_content: str
    pages: List[DocumentPage]
    tables: List[DocumentTable]
    metadata: Dict[str, Any]
    features: DocumentFeatures
    model_used: str
    confidence_scores: Dict[str, float]
    key_value_pairs: List[Dict[str, str]]
    page_count: int
    processing_time: float
    figures: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]


class DocumentExtractor:
    """
    Main interface for document text extraction using Azure Document Intelligence.

    Provides automatic model selection, document analysis, and text extraction
    capabilities with support for multiple document formats.
    """

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize document extractor.

        Args:
            endpoint: Azure Document Intelligence endpoint (uses env var if None)
            key: Azure Document Intelligence API key (uses env var if None)
        """
        self.endpoint = endpoint
        self.key = key

        # Initialize client if credentials provided
        if endpoint and key:
            initialize_client(endpoint, key)
        elif not is_client_initialized():
            # Try to initialize with environment variables
            try:
                initialize_client()
            except Exception as e:
                logger.warning(f"Could not initialize client with environment variables: {e}")

        logger.info("DocumentExtractor initialized")

    def extract_text(
        self, file_path: Union[str, Path], force_model: Optional[str] = None, auto_detect_model: bool = True
    ) -> ExtractionResult:
        """
        Extract text from a document with automatic model selection.

        Args:
            file_path: Path to the document file
            force_model: Force a specific model type (overrides auto-detection)
            auto_detect_model: Whether to use automatic model detection

        Returns:
            ExtractionResult with extracted text and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
            RuntimeError: If extraction fails
        """
        try:
            # Validate and normalize file path
            file_path = self._validate_file(file_path)

            logger.info(f"Starting text extraction for file: {file_path}")

            # Extract document features for model selection
            features = self._extract_document_features(file_path)

            if features.error:
                logger.warning(f"Feature extraction had errors: {features.error}")

            # Determine model type
            if force_model:
                model_type = force_model
                logger.info(f"Using forced model type: {model_type}")
            elif auto_detect_model:
                model_type = self._determine_model_type(features)
                logger.info(f"Auto-detected model type: {model_type}")
            else:
                model_type = "layout"  # Default fallback
                logger.info(f"Using default model type: {model_type}")

            # Process document with selected model
            analysis_result, metadata = self._process_document(file_path, model_type)

            # Extract comprehensive document structure
            document_structure = self._extract_document_structure(analysis_result)

            # Calculate confidence scores (placeholder for actual model confidence)
            confidence_scores = {model_type: 1.0}

            result = ExtractionResult(
                text_content=document_structure["text_content"],
                pages=document_structure["pages"],
                tables=document_structure["tables"],
                metadata=metadata,
                features=features,
                model_used=model_type,
                confidence_scores=confidence_scores,
                key_value_pairs=document_structure["key_value_pairs"],
                page_count=len(document_structure["pages"]),
                processing_time=metadata.get("processing_time_seconds", 0.0),
                figures=document_structure["figures"],
                paragraphs=document_structure["paragraphs"],
            )

            logger.info(f"Text extraction completed: {len(result.text_content)} characters extracted")
            return result

        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            raise

    def analyze_document(self, file_path: Union[str, Path]) -> DocumentFeatures:
        """
        Analyze document to extract features without full processing.

        Args:
            file_path: Path to the document file

        Returns:
            DocumentFeatures with analysis results
        """
        file_path = self._validate_file(file_path)
        return self._extract_document_features(file_path)

    def get_supported_models(self) -> List[str]:
        """Get list of supported model types."""
        return list(MODEL_TYPES.keys())

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return SUPPORTED_FILE_EXTENSIONS.copy()

    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file exists and has a supported extension.

        Args:
            file_path: Path to the file to validate

        Returns:
            Path object of the validated file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file has an unsupported extension
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if not validate_file_extension(file_path.suffix):
            logger.error(f"Unsupported file type: {file_path.suffix}")
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. " f"Supported types: {SUPPORTED_FILE_EXTENSIONS}"
            )

        return file_path

    def _extract_document_features(self, file_path: Path) -> DocumentFeatures:
        """
        Extract basic features from a document using layout model.

        Args:
            file_path: Path to the document file

        Returns:
            DocumentFeatures with extracted information
        """
        try:
            logger.info(f"Extracting features from document: {file_path}")

            # Get client
            client = get_document_client(self.endpoint, self.key)

            # Read document
            with open(file_path, "rb") as f:
                document_bytes = f.read()

            # Set content type
            content_type = get_content_type(file_path.suffix)

            # Use layout model for feature extraction
            logger.info("Using layout model for feature extraction")
            start_time = datetime.now()

            analysis_operation = client.begin_analyze_document(
                model_id=get_model_id("layout"), body=document_bytes, content_type=content_type
            )

            analysis_result = analysis_operation.result()
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Feature extraction completed in {processing_time:.2f} seconds")

            # Extract features
            text_content = getattr(analysis_result, "content", "")
            pages = getattr(analysis_result, "pages", [])

            # Extract first page text
            first_page_text = ""
            if pages:
                page = pages[0]
                if hasattr(page, "lines"):
                    first_page_text = "\n".join([line.content for line in page.lines if hasattr(line, "content")])

            # Extract key-value pairs
            key_value_pairs = []
            key_phrases = []
            if hasattr(analysis_result, "key_value_pairs") and analysis_result.key_value_pairs:
                for kv in analysis_result.key_value_pairs:
                    key = getattr(kv.key, "content", "") if hasattr(kv, "key") and kv.key else ""
                    value = getattr(kv.value, "content", "") if hasattr(kv, "value") and kv.value else ""
                    if key:
                        key_value_pairs.append({"key": key, "value": value})
                        key_phrases.append(key.lower())

            # Check for tables
            tables = getattr(analysis_result, "tables", [])
            has_tables = len(tables) > 0

            features = DocumentFeatures(
                file_name=file_path.name,
                file_size=os.path.getsize(file_path),
                file_extension=file_path.suffix.lower(),
                text_content=text_content,
                page_count=len(pages),
                key_value_pairs=key_value_pairs,
                key_phrases=key_phrases,
                first_page_text=first_page_text,
                has_tables=has_tables,
                table_count=len(tables),
                processing_time=processing_time,
            )

            logger.info(f"Document features extracted: {len(text_content)} chars, {len(pages)} pages")
            return features

        except HttpResponseError as error:
            error_message = str(error)
            logger.error(f"Azure Document Intelligence error during feature extraction: {error_message}")

            # Return basic features even on API error
            return DocumentFeatures(
                file_name=file_path.name,
                file_size=os.path.getsize(file_path) if file_path.exists() else 0,
                file_extension=file_path.suffix.lower(),
                text_content="",
                page_count=0,
                key_value_pairs=[],
                key_phrases=[],
                first_page_text="",
                has_tables=False,
                table_count=0,
                processing_time=0.0,
                error=error_message,
            )
        except Exception as e:
            logger.error(f"Error extracting document features: {e}")

            return DocumentFeatures(
                file_name=file_path.name,
                file_size=os.path.getsize(file_path) if file_path.exists() else 0,
                file_extension=file_path.suffix.lower(),
                text_content="",
                page_count=0,
                key_value_pairs=[],
                key_phrases=[],
                first_page_text="",
                has_tables=False,
                table_count=0,
                processing_time=0.0,
                error=str(e),
            )

    def _determine_model_type(self, features: DocumentFeatures) -> str:
        """
        Determine the most appropriate model type based on document features.

        Args:
            features: Document features extracted from initial analysis

        Returns:
            Recommended model type
        """
        try:
            logger.info("Determining optimal document model type")

            # Extract relevant features
            text = features.text_content.lower()
            features.first_page_text.lower()
            key_phrases = features.key_phrases
            file_size = features.file_size
            has_tables = features.has_tables
            page_count = features.page_count

            # Initialize confidence scores
            confidence_scores = {doc_type: 0 for doc_type in DOCUMENT_TYPE_INDICATORS.keys()}

            # Get configuration
            config = MODEL_DETECTION_CONFIG
            base_threshold = config["base_confidence_threshold"]
            min_gap = config["min_confidence_gap"]

            # Calculate scores based on term presence
            for doc_type, indicators in DOCUMENT_TYPE_INDICATORS.items():
                score = 0
                detected_terms = []

                for term, weight in indicators.items():
                    # Check in full text
                    if term in text:
                        score += weight
                        detected_terms.append(term)

                    # Check in key phrases (higher weight)
                    if any(term in key.lower() for key in key_phrases):
                        score += weight * 1.5
                        if term not in detected_terms:
                            detected_terms.append(term)

                confidence_scores[doc_type] = score
                logger.debug(f"{doc_type} score: {score}, terms: {detected_terms}")

            # Apply structural factors
            size_thresholds = config["file_size_thresholds"]

            if file_size < size_thresholds["small"]:
                confidence_scores["business_card"] += 15
                confidence_scores["receipt"] += 10
                confidence_scores["id"] += 5
            elif file_size < size_thresholds["medium"]:
                confidence_scores["receipt"] += 15
                confidence_scores["business_card"] += 5
                confidence_scores["id"] += 10
            elif file_size > size_thresholds["large"]:
                confidence_scores["invoice"] += 10
                confidence_scores["tax"] += 10
                confidence_scores["receipt"] -= 5
                confidence_scores["business_card"] -= 10

            # Page count factors
            if page_count == 1:
                confidence_scores["receipt"] += 10
                confidence_scores["business_card"] += 15
                confidence_scores["id"] += 10
            elif page_count >= 3:
                confidence_scores["invoice"] += 10
                confidence_scores["tax"] += 10
                confidence_scores["receipt"] -= 10
                confidence_scores["business_card"] -= 15

            # Table presence
            if has_tables:
                confidence_scores["invoice"] += 15
                confidence_scores["tax"] += 10
                confidence_scores["receipt"] += 5

            # Find highest scoring type
            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            highest_type, highest_score = sorted_scores[0]
            second_type, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0)

            confidence_gap = highest_score - second_score

            logger.info(f"Model detection scores: {dict(sorted_scores)}")
            logger.info(f"Highest: {highest_type} ({highest_score}), Gap: {confidence_gap}")

            # Decision logic
            if highest_score >= base_threshold and confidence_gap >= min_gap:
                logger.info(f"Selected model type '{highest_type}' with confidence {highest_score}")
                return highest_type

            # Fall back to layout for ambiguous cases
            logger.info("Confidence too low or ambiguous, falling back to 'layout'")
            return "layout"

        except Exception as e:
            logger.error(f"Error in model type determination: {e}")
            return "layout"

    def _process_document(self, file_path: Path, model_type: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Process a document using Azure Document Intelligence.

        Args:
            file_path: Path to the document file
            model_type: Type of model to use for analysis

        Returns:
            Tuple containing analysis result and metadata
        """
        try:
            logger.info(f"Processing document: {file_path} with model: {model_type}")

            # Get client
            client = get_document_client(self.endpoint, self.key)

            # Read document
            with open(file_path, "rb") as f:
                document_bytes = f.read()

            # Get model ID and content type
            model_id = get_model_id(model_type)
            content_type = get_content_type(file_path.suffix)

            logger.info(f"Using model_id: {model_id}, content_type: {content_type}")

            # Process document
            start_time = datetime.now()
            analysis_operation = client.begin_analyze_document(
                model_id=model_id, body=document_bytes, content_type=content_type
            )

            analysis_result = analysis_operation.result()
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix[1:],
                "file_size_bytes": len(document_bytes),
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time_seconds": processing_time,
                "model_used": model_id,
                "model_type": model_type,
            }

            logger.info(f"Document processed successfully in {processing_time:.2f} seconds")
            return analysis_result, metadata

        except HttpResponseError as error:
            logger.error(f"Azure Document Intelligence error: {error}")
            raise
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    def _extract_document_structure(self, analysis_result: Any) -> Dict[str, Any]:
        """
        Extract comprehensive document structure from Azure Document Intelligence result.

        Args:
            analysis_result: Azure Document Intelligence analysis result

        Returns:
            Dictionary with structured document data
        """
        try:
            logger.info("Extracting comprehensive document structure")

            # Extract text content
            text_content = getattr(analysis_result, "content", "")

            # Extract pages with detailed information
            pages = []
            raw_pages = getattr(analysis_result, "pages", [])

            for page in raw_pages:
                page_data = DocumentPage(
                    page_number=getattr(page, "page_number", 0),
                    width=getattr(page, "width", 0.0),
                    height=getattr(page, "height", 0.0),
                    unit=getattr(page, "unit", "pixel"),
                    angle=getattr(page, "angle", 0.0),
                    lines=self._extract_lines(page),
                    words=self._extract_words(page),
                    selection_marks=self._extract_selection_marks(page),
                )
                pages.append(page_data)

            # Extract tables with full structure
            tables = []
            raw_tables = getattr(analysis_result, "tables", [])

            for table in raw_tables:
                table_data = DocumentTable(
                    row_count=getattr(table, "row_count", 0),
                    column_count=getattr(table, "column_count", 0),
                    cells=self._extract_table_cells(table),
                    bounding_regions=self._extract_bounding_regions(table),
                    spans=self._extract_spans(table),
                )
                tables.append(table_data)

            # Extract key-value pairs
            key_value_pairs = []
            if hasattr(analysis_result, "key_value_pairs") and analysis_result.key_value_pairs:
                for kv in analysis_result.key_value_pairs:
                    key = getattr(kv.key, "content", "") if hasattr(kv, "key") and kv.key else ""
                    value = getattr(kv.value, "content", "") if hasattr(kv, "value") and kv.value else ""
                    if key:
                        key_value_pairs.append({"key": key, "value": value})

            # Extract figures (if available)
            figures = []
            if hasattr(analysis_result, "figures") and analysis_result.figures:
                for figure in analysis_result.figures:
                    figure_data = {
                        "id": getattr(figure, "id", ""),
                        "bounding_regions": self._extract_bounding_regions(figure),
                        "spans": self._extract_spans(figure),
                        "elements": getattr(figure, "elements", []),
                    }
                    figures.append(figure_data)

            # Extract paragraphs (if available)
            paragraphs = []
            if hasattr(analysis_result, "paragraphs") and analysis_result.paragraphs:
                for paragraph in analysis_result.paragraphs:
                    paragraph_data = {
                        "content": getattr(paragraph, "content", ""),
                        "bounding_regions": self._extract_bounding_regions(paragraph),
                        "spans": self._extract_spans(paragraph),
                        "role": getattr(paragraph, "role", None),
                    }
                    paragraphs.append(paragraph_data)

            structure = {
                "text_content": text_content,
                "pages": pages,
                "tables": tables,
                "key_value_pairs": key_value_pairs,
                "figures": figures,
                "paragraphs": paragraphs,
            }

            logger.info(
                f"Document structure extracted: {len(pages)} pages, {len(tables)} tables, {len(figures)} figures"
            )
            return structure

        except Exception as e:
            logger.error(f"Error extracting document structure: {e}")
            # Return minimal structure on error
            return {
                "text_content": getattr(analysis_result, "content", ""),
                "pages": [],
                "tables": [],
                "key_value_pairs": [],
                "figures": [],
                "paragraphs": [],
            }

    def _extract_lines(self, page: Any) -> List[Dict[str, Any]]:
        """Extract line information from a page."""
        lines = []
        if hasattr(page, "lines") and page.lines:
            for line in page.lines:
                line_data = {
                    "content": getattr(line, "content", ""),
                    "polygon": getattr(line, "polygon", []),
                    "spans": self._extract_spans(line),
                }
                lines.append(line_data)
        return lines

    def _extract_words(self, page: Any) -> List[Dict[str, Any]]:
        """Extract word information from a page."""
        words = []
        if hasattr(page, "words") and page.words:
            for word in page.words:
                word_data = {
                    "content": getattr(word, "content", ""),
                    "polygon": getattr(word, "polygon", []),
                    "confidence": getattr(word, "confidence", 0.0),
                    "spans": self._extract_spans(word),
                }
                words.append(word_data)
        return words

    def _extract_selection_marks(self, page: Any) -> List[Dict[str, Any]]:
        """Extract selection marks (checkboxes, radio buttons) from a page."""
        selection_marks = []
        if hasattr(page, "selection_marks") and page.selection_marks:
            for mark in page.selection_marks:
                mark_data = {
                    "state": getattr(mark, "state", ""),
                    "polygon": getattr(mark, "polygon", []),
                    "confidence": getattr(mark, "confidence", 0.0),
                    "spans": self._extract_spans(mark),
                }
                selection_marks.append(mark_data)
        return selection_marks

    def _extract_table_cells(self, table: Any) -> List[Dict[str, Any]]:
        """Extract cell information from a table."""
        cells = []
        if hasattr(table, "cells") and table.cells:
            for cell in table.cells:
                cell_data = {
                    "content": getattr(cell, "content", ""),
                    "row_index": getattr(cell, "row_index", 0),
                    "column_index": getattr(cell, "column_index", 0),
                    "row_span": getattr(cell, "row_span", 1),
                    "column_span": getattr(cell, "column_span", 1),
                    "kind": getattr(cell, "kind", "content"),
                    "bounding_regions": self._extract_bounding_regions(cell),
                    "spans": self._extract_spans(cell),
                }
                cells.append(cell_data)
        return cells

    def _extract_bounding_regions(self, element: Any) -> List[Dict[str, Any]]:
        """Extract bounding region information from an element."""
        regions = []
        if hasattr(element, "bounding_regions") and element.bounding_regions:
            for region in element.bounding_regions:
                region_data = {
                    "page_number": getattr(region, "page_number", 1),
                    "polygon": getattr(region, "polygon", []),
                }
                regions.append(region_data)
        return regions

    def _extract_spans(self, element: Any) -> List[Dict[str, Any]]:
        """Extract span information from an element."""
        spans = []
        if hasattr(element, "spans") and element.spans:
            for span in element.spans:
                span_data = {"offset": getattr(span, "offset", 0), "length": getattr(span, "length", 0)}
                spans.append(span_data)
        return spans
