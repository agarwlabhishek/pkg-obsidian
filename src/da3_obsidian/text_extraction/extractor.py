"""
Main text extraction interface combining PDF and image processing capabilities.
"""

import logging
from typing import Optional, Tuple

from .image_processor import ImageProcessor
from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Main interface for text extraction from various document types.

    Supports:
    - PDF documents with embedded text and images
    - Image files using OCR
    - Quality validation and confidence scoring
    """

    def __init__(self, language: str = "en", gibberish_threshold: float = 0.7):
        """
        Initialize text extractor.

        Args:
            language: Language code for OCR ('en', 'es', 'fr', 'it')
            gibberish_threshold: Threshold for gibberish detection (0.0-1.0)
        """
        self.language = language
        self.gibberish_threshold = gibberish_threshold

        # Initialize processors
        self.pdf_processor = PDFProcessor(language, gibberish_threshold)
        self.image_processor = ImageProcessor(language)

        logger.info(f"TextExtractor initialized for language: {language}")

    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, confidence_score)

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For processing errors
        """
        try:
            logger.info(f"Starting PDF extraction: {pdf_path}")
            text, confidence = self.pdf_processor.extract_from_pdf(pdf_path)
            logger.info(f"PDF extraction completed with {confidence} confidence")
            return text, confidence
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    def extract_from_image(self, image_path: str, validate_quality: bool = True) -> str:
        """
        Extract text from image file using OCR.

        Args:
            image_path: Path to image file
            validate_quality: Whether to validate OCR quality

        Returns:
            Extracted text

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: For processing errors
        """
        try:
            logger.info(f"Starting image OCR: {image_path}")
            text = self.image_processor.extract_from_file(image_path)

            if validate_quality:
                is_valid = self.image_processor.quality_checker.is_text_valid(
                    text, self.language, self.gibberish_threshold
                )
                if not is_valid:
                    logger.warning("OCR text quality is low")

            logger.info(f"Image OCR completed, {len(text)} characters extracted")
            return text

        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            raise

    def get_text_quality_score(self, text: str) -> Tuple[float, float]:
        """
        Get quality metrics for extracted text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (gibberish_set_ratio, gibberish_tokens_ratio)
        """
        try:
            return self.image_processor.quality_checker.calculate_gibberish_ratio(text, self.language)
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return float("inf"), float("inf")

    def is_text_high_quality(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: Text to validate

        Returns:
            True if text quality is acceptable
        """
        return self.image_processor.quality_checker.is_text_valid(text, self.language, self.gibberish_threshold)

    def set_language(self, language: str) -> None:
        """
        Change the processing language.

        Args:
            language: New language code ('en', 'es', 'fr', 'it')
        """
        if language not in ["en", "es", "fr", "it"]:
            raise ValueError(f"Unsupported language: {language}")

        self.language = language
        self.pdf_processor.language = language
        self.image_processor.language = language
        self.image_processor.tesseract_lang = self.image_processor.TESSERACT_LANG_MAPPING.get(language, "eng")

        logger.info(f"Language changed to: {language}")

    def set_gibberish_threshold(self, threshold: float) -> None:
        """
        Change the gibberish detection threshold.

        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.gibberish_threshold = threshold
        self.pdf_processor.gibberish_threshold = threshold

        logger.info(f"Gibberish threshold changed to: {threshold}")

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return ["en", "es", "fr", "it"]

    def get_current_config(self) -> dict:
        """Get current extractor configuration."""
        return {
            "language": self.language,
            "gibberish_threshold": self.gibberish_threshold,
            "supported_languages": self.get_supported_languages(),
        }
