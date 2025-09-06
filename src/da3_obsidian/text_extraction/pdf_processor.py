"""
PDF processing module for extracting text, tables, and images from PDF documents.
"""

import hashlib
import io
import logging
import os
from typing import List, Optional, Set, Tuple

import numpy as np
import pdfplumber
from PIL import Image

from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction with OCR support for embedded images."""

    def __init__(self, language: str = "en", gibberish_threshold: float = 0.7):
        """
        Initialize PDF processor.

        Args:
            language: Language code for OCR ('en', 'es', 'fr', 'it')
            gibberish_threshold: Threshold for gibberish detection (0.0-1.0)
        """
        self.language = language
        self.gibberish_threshold = gibberish_threshold
        self.image_processor = ImageProcessor(language=language)
        self._seen_image_hashes: Set[str] = set()

    def table_to_markdown(self, table: List[List]) -> str:
        """Convert a table (list of lists) to Markdown format."""
        try:
            if not table:
                return ""

            markdown = ""
            for i, row in enumerate(table):
                row = [str(item) if item is not None else "" for item in row]
                markdown += "| " + " | ".join(row) + " |\n"
                if i == 0:
                    markdown += "| " + " | ".join(["---"] * len(row)) + " |\n"
            return markdown
        except Exception as e:
            logger.error(f"Error converting table to Markdown: {e}")
            return ""

    def _adjust_bbox_to_page_limits(
        self, bbox: Tuple[float, float, float, float], page_width: float, page_height: float
    ) -> Tuple[float, float, float, float]:
        """Adjust a bounding box to ensure it stays within page limits."""
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(x0, page_width))
        x1 = max(0, min(x1, page_width))
        y0 = max(0, min(y0, page_height))
        y1 = max(0, min(y1, page_height))
        return (x0, y0, x1, y1)

    def _get_sub_pages_with_tables(self, page):
        """Split the page into sub-regions based on detected tables."""
        try:
            tables = page.find_tables()
            tables = sorted(tables, key=lambda x: x.bbox[1])
            sub_pages = []
            prev_bottom = 0

            for table in tables:
                if prev_bottom < table.bbox[1]:
                    sub_pages.append((prev_bottom, table.bbox[1]))
                sub_pages.append(table.bbox)
                prev_bottom = table.bbox[3]

            if prev_bottom < page.height:
                sub_pages.append((prev_bottom, page.height))

            return sub_pages
        except Exception as e:
            logger.error(f"Error getting sub-pages with tables: {e}")
            return []

    def _extract_text_from_sub_pages(self, page, sub_pages: List) -> Tuple[str, str]:
        """Extract text and tables from each sub-page region."""
        page_text = ""
        digital_text = ""
        page_width = page.width
        page_height = page.height

        for bbox in sub_pages:
            if len(bbox) == 4:
                # Table region
                try:
                    adjusted_bbox = self._adjust_bbox_to_page_limits(bbox, page_width, page_height)
                    sub_page = page.crop(adjusted_bbox)
                    sub_table = sub_page.extract_table()

                    if sub_table:
                        markdown_table = self.table_to_markdown(sub_table)
                        page_text += markdown_table
                        digital_text += markdown_table

                    text = sub_page.extract_text()
                    if text:
                        page_text += text
                        digital_text += text

                except Exception as e:
                    logger.error(f"Error processing table region: {e}")
            else:
                # Non-table region
                try:
                    adjusted_bbox = self._adjust_bbox_to_page_limits(
                        (0, bbox[0], page.width, bbox[1]), page_width, page_height
                    )
                    sub_page = page.crop(adjusted_bbox)
                    text = sub_page.extract_text()
                    if text:
                        page_text += text
                        digital_text += text
                except Exception as e:
                    logger.error(f"Error processing non-table region: {e}")

        return page_text, digital_text

    def _crop_visual_to_numpy(self, page, visual, resolution: int = 300) -> np.ndarray:
        """Crop the PDF page to the bounding box of the visual object."""
        page_width = page.width
        page_height = page.height
        original_bbox = (visual["x0"], page_height - visual["y1"], visual["x1"], page_height - visual["y0"])
        adjusted_bbox = self._adjust_bbox_to_page_limits(original_bbox, page_width, page_height)

        cropped_page = page.crop(adjusted_bbox)
        image_obj = cropped_page.to_image(resolution=resolution)
        pil_image = image_obj.original.convert("L")
        return np.array(pil_image)

    def _hash_image_array(self, np_image: np.ndarray) -> str:
        """Create a hash from numpy image array for duplicate detection."""
        pil_image = Image.fromarray(np_image)
        with io.BytesIO() as output:
            pil_image.save(output, format="PNG")
            img_bytes = output.getvalue()
        return hashlib.sha256(img_bytes).hexdigest()

    def _process_page_visuals(self, page, page_num: int) -> Tuple[str, str, int]:
        """Process all visual elements (images) on a page."""
        page_text = ""
        image_ocr_text = ""
        processed_images = 0
        max_images = 50

        visuals = page.images
        for visual in visuals:
            if processed_images >= max_images:
                logger.warning("Maximum image limit reached. Stopping further processing.")
                break

            try:
                # Extract image as numpy array
                np_image = self._crop_visual_to_numpy(page, visual)

                # Check for duplicates
                image_hash = self._hash_image_array(np_image)
                if image_hash in self._seen_image_hashes:
                    logger.info(f"Duplicate image on page {page_num}, skipping.")
                    continue
                self._seen_image_hashes.add(image_hash)

                # Perform OCR
                image_text = self.image_processor.extract_from_array(np_image)
                image_ocr_text += image_text

                # Validate quality and add to page text if good
                if self.image_processor.quality_checker.is_text_valid(
                    image_text, self.language, self.gibberish_threshold
                ):
                    page_text += image_text
                else:
                    logger.info("Skipping low-quality OCR text from image.")

                processed_images += 1

            except Exception as e:
                logger.error(f"Error processing image on page {page_num}: {e}")

        return page_text, image_ocr_text, processed_images

    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from PDF file with OCR support for images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        document_text = ""
        digital_text = ""
        image_ocr_text = ""
        processed_images = 0
        max_images = 50

        # Reset seen images for new document
        self._seen_image_hashes.clear()

        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF: {pdf_path}, pages: {len(pdf.pages)}")

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text and tables from page regions
                    sub_pages = self._get_sub_pages_with_tables(page)
                    page_text, page_digital = self._extract_text_from_sub_pages(page, sub_pages)
                    digital_text += page_digital

                    # Process images on page
                    img_text, img_ocr, img_count = self._process_page_visuals(page, page_num)
                    page_text += img_text
                    image_ocr_text += img_ocr
                    processed_images += img_count

                    # Add page content with separator
                    if len(page_text) > 10:
                        document_text += page_text + "\n--- PAGE BREAK ---\n"

                    # Stop if too many images processed
                    if processed_images >= max_images:
                        document_text += "\nWARNING: Processing terminated early due to document length!\n"
                        break

            # Calculate confidence score
            confidence = self.image_processor.quality_checker.determine_confidence(
                digital_text, image_ocr_text, self.language, self.gibberish_threshold
            )

            logger.info(f"PDF extraction completed. Confidence: {confidence}")
            return document_text, confidence

        except Exception as e:
            logger.error(f"Failed to extract text from PDF '{pdf_path}': {e}")
            raise
