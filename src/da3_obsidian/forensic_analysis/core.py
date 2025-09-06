"""
Core forensic analysis functionality.
"""

import datetime
import hashlib
import logging
import os
import traceback
from typing import Any, Dict, Optional

from .config import ForensicConfig
from .image_analyzer import ImageAnalyzer
from .pdf_analyzer import PDFAnalyzer

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Core document analyzer that coordinates all forensic analysis components."""

    def __init__(self, config: ForensicConfig):
        """
        Initialize document analyzer with configuration.

        Args:
            config: Forensic analysis configuration
        """
        self.config = config

        # Initialize component analyzers
        self.pdf_analyzer = PDFAnalyzer(config)
        self.image_analyzer = ImageAnalyzer(config)

        # Configure logging
        log_level = getattr(logging, config.get("General", "log_level", "INFO").upper(), logging.INFO)
        logger.setLevel(log_level)

        logger.info("DocumentAnalyzer initialized")

    def analyze_document(self, file_path: str, trusted_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a document for potential fraud indicators.

        Args:
            file_path: Path to the document file
            trusted_hash: Optional trusted hash for integrity verification

        Returns:
            Dictionary with comprehensive analysis results

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        logger.info(f"Starting document analysis: {file_path}")

        # Initialize results structure
        results = {
            "file_info": {},
            "metadata_analysis": {},
            "integrity_analysis": {},
            "text_analysis": {},
            "image_analysis": {},
            "summary": {"potential_fraud_indicators": [], "confidence": {}},
        }

        try:
            # Validate file existence
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Extract basic file information
            self._extract_file_info(file_path, results, trusted_hash)

            # Determine file type and dispatch to appropriate analyzer
            file_ext = os.path.splitext(file_path)[1].lower()

            # Get supported formats
            supported_extensions = {
                ".pdf": "PDF document",
                ".jpg": "JPEG image",
                ".jpeg": "JPEG image",
                ".png": "PNG image",
                ".bmp": "BMP image",
                ".tiff": "TIFF image",
                ".tif": "TIFF image",
            }

            if file_ext not in supported_extensions:
                raise ValueError(f"Unsupported file format: {file_ext}. Supported: {list(supported_extensions.keys())}")

            logger.info(f"Detected file type: {supported_extensions[file_ext]}")

            if file_ext == ".pdf":
                self._analyze_pdf_document(file_path, results)
            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                self._analyze_image_document(file_path, results)

            # Calculate final risk score
            self._calculate_risk_score(results)

            # Add completion timestamp
            results["summary"]["analysis_completed"] = datetime.datetime.now().isoformat()
            results["summary"]["analysis_successful"] = True

            logger.info(f"Analysis completed successfully: {results['summary']['risk_level']}")
            return results

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

            # Add error information to results
            results["summary"]["analysis_error"] = {
                "message": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            results["summary"]["analysis_completed"] = datetime.datetime.now().isoformat()
            results["summary"]["analysis_successful"] = False

            return results

    def _extract_file_info(self, file_path: str, results: Dict[str, Any], trusted_hash: Optional[str] = None) -> None:
        """Extract basic file information and calculate hashes."""
        try:
            file_size = os.path.getsize(file_path)
            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            file_created = datetime.datetime.fromtimestamp(os.path.getctime(file_path))

            # Calculate file hashes
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                md5_hash = hashlib.md5(file_bytes).hexdigest()
                sha256_hash = hashlib.sha256(file_bytes).hexdigest()

            results["file_info"] = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_extension": os.path.splitext(file_path)[1].lower(),
                "file_size_bytes": file_size,
                "file_size_human": self._format_file_size(file_size),
                "file_modified": file_modified.isoformat(),
                "file_created": file_created.isoformat(),
                "analysis_timestamp": datetime.datetime.now().isoformat(),
            }

            results["integrity_analysis"]["file_hashes"] = {
                "md5": md5_hash,
                "sha256": sha256_hash,
                "matches_trusted_hash": trusted_hash == sha256_hash if trusted_hash else None,
            }

            logger.info(f"File info extracted: {os.path.basename(file_path)}, {self._format_file_size(file_size)}")
            logger.info(f"File hashes calculated: md5={md5_hash[:8]}..., sha256={sha256_hash[:8]}...")

        except Exception as e:
            logger.error(f"Error extracting file info: {e}")
            raise

    def _analyze_pdf_document(self, file_path: str, results: Dict[str, Any]) -> None:
        """Analyze PDF document using PDF analyzer."""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            self.pdf_analyzer.analyze_pdf(file_path, file_bytes, results)

        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            raise

    def _analyze_image_document(self, file_path: str, results: Dict[str, Any]) -> None:
        """Analyze image document using image analyzer."""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            self.image_analyzer.analyze_image(file_path, file_bytes, results)

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise

    def _calculate_risk_score(self, results: Dict[str, Any]) -> None:
        """Calculate overall risk score based on analysis findings."""
        if "summary" not in results:
            results["summary"] = {"potential_fraud_indicators": []}

        # Count fraud indicators
        fraud_indicator_count = len(results["summary"]["potential_fraud_indicators"])

        # Calculate risk score (0-100) based on the number of indicators
        max_indicators = 20  # Normalization factor
        risk_score = min(100, int((fraud_indicator_count / max_indicators) * 100))

        # Determine risk level
        if risk_score < 20:
            risk_level = "Low"
        elif risk_score < 50:
            risk_level = "Medium"
        elif risk_score < 80:
            risk_level = "High"
        else:
            risk_level = "Very High"

        results["summary"]["risk_score"] = risk_score
        results["summary"]["risk_level"] = risk_level

        logger.info(f"Risk assessment: {risk_level} ({risk_score}/100) with {fraud_indicator_count} indicators")

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
