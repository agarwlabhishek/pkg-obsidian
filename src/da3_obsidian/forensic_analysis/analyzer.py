"""
Main forensic analyzer interface for document fraud detection.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import ForensicConfig
from .core import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ForensicAnalysisResult:
    """Result of forensic analysis with structured access to findings."""

    file_path: str
    risk_score: int
    risk_level: str
    fraud_indicators: list

    # Analysis sections
    metadata_issues: list
    image_issues: list
    structure_issues: list
    text_issues: list

    # Convenience properties
    has_metadata_issues: bool
    has_image_manipulation: bool
    has_structure_anomalies: bool
    has_text_inconsistencies: bool

    # Raw analysis data
    raw_results: Dict[str, Any]

    @property
    def file_info(self):
        """Access to file info from raw results."""
        return self.raw_results.get("file_info", {})

    @property
    def is_high_risk(self) -> bool:
        """Check if result indicates high risk."""
        return self.risk_level in ["High", "Very High"]

    @property
    def total_issues(self) -> int:
        """Get total number of fraud indicators."""
        return len(self.fraud_indicators)

    def get_issues_by_category(self) -> Dict[str, list]:
        """Categorize fraud indicators by analysis type."""
        categories = {
            "metadata": self.metadata_issues,
            "image_manipulation": self.image_issues,
            "structure_anomalies": self.structure_issues,
            "text_inconsistencies": self.text_issues,
        }
        return categories

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        issues_by_category = self.get_issues_by_category()

        return {
            "file_name": os.path.basename(self.file_path),
            "file_size": self.file_info.get("file_size_human", "Unknown"),
            "risk_assessment": {
                "score": self.risk_score,
                "level": self.risk_level,
                "is_high_risk": self.is_high_risk,
                "total_indicators": self.total_issues,
            },
            "findings_by_category": {k: len(v) for k, v in issues_by_category.items()},
            "has_issues": {
                "metadata": self.has_metadata_issues,
                "images": self.has_image_manipulation,
                "structure": self.has_structure_anomalies,
                "text": self.has_text_inconsistencies,
            },
        }

    @classmethod
    def from_dict(cls, results: Dict[str, Any]) -> "ForensicAnalysisResult":
        """Create ForensicAnalysisResult from raw analysis dictionary."""
        summary = results.get("summary", {})
        fraud_indicators = summary.get("potential_fraud_indicators", [])

        # Categorize issues by keywords
        metadata_issues = []
        image_issues = []
        structure_issues = []
        text_issues = []

        for indicator in fraud_indicators:
            indicator_lower = indicator.lower()

            if any(
                word in indicator_lower for word in ["metadata", "date", "software", "missing", "creator", "producer"]
            ):
                metadata_issues.append(indicator)
            elif any(
                word in indicator_lower
                for word in ["image", "manipulation", "ela", "ghost", "copy-move", "noise", "shadow", "edge", "cfa"]
            ):
                image_issues.append(indicator)
            elif any(
                word in indicator_lower
                for word in [
                    "structure",
                    "javascript",
                    "layer",
                    "update",
                    "signature",
                    "encrypted",
                    "embedded",
                    "filter",
                ]
            ):
                structure_issues.append(indicator)
            elif any(word in indicator_lower for word in ["font", "text"]):
                text_issues.append(indicator)
            else:
                # Default to metadata if unclear
                metadata_issues.append(indicator)

        return cls(
            file_path=results.get("file_info", {}).get("file_path", ""),
            risk_score=summary.get("risk_score", 0),
            risk_level=summary.get("risk_level", "Unknown"),
            fraud_indicators=fraud_indicators,
            metadata_issues=metadata_issues,
            image_issues=image_issues,
            structure_issues=structure_issues,
            text_issues=text_issues,
            has_metadata_issues=len(metadata_issues) > 0,
            has_image_manipulation=len(image_issues) > 0,
            has_structure_anomalies=len(structure_issues) > 0,
            has_text_inconsistencies=len(text_issues) > 0,
            raw_results=results,
        )


class ForensicAnalyzer:
    """
    Main interface for document forensic analysis.

    Provides comprehensive fraud detection capabilities including:
    - PDF metadata and structure analysis
    - Image manipulation detection
    - Text consistency verification
    - Risk assessment and scoring
    """

    def __init__(self, config: Optional[ForensicConfig] = None):
        """
        Initialize forensic analyzer.

        Args:
            config: Optional configuration object. Uses defaults if None.
        """
        self.config = config or ForensicConfig()

        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Invalid configuration: {e}")
            raise

        # Initialize core analyzer
        self.analyzer = DocumentAnalyzer(self.config)

        logger.info("ForensicAnalyzer initialized")

    def analyze_document(self, file_path: str, trusted_hash: Optional[str] = None) -> ForensicAnalysisResult:
        """
        Analyze a document for potential fraud indicators.

        Args:
            file_path: Path to the document file to analyze
            trusted_hash: Optional SHA256 hash for integrity verification

        Returns:
            ForensicAnalysisResult with structured findings

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            Exception: If the analysis fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Starting forensic analysis: {file_path}")

        try:
            # Run the core analysis
            raw_results = self.analyzer.analyze_document(file_path, trusted_hash)

            # Convert to structured result
            result = ForensicAnalysisResult.from_dict(raw_results)

            logger.info(f"Analysis complete: {result.risk_level} risk ({result.risk_score}/100)")
            return result

        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            raise

    def analyze_pdf(self, file_path: str) -> ForensicAnalysisResult:
        """
        Analyze a PDF document specifically.

        Args:
            file_path: Path to PDF file

        Returns:
            ForensicAnalysisResult with PDF-specific findings
        """
        if not file_path.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF document")

        return self.analyze_document(file_path)

    def analyze_image(self, file_path: str) -> ForensicAnalysisResult:
        """
        Analyze an image file specifically.

        Args:
            file_path: Path to image file

        Returns:
            ForensicAnalysisResult with image-specific findings
        """
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        if not any(file_path.lower().endswith(fmt) for fmt in supported_formats):
            raise ValueError(f"File must be an image. Supported formats: {supported_formats}")

        return self.analyze_document(file_path)

    def batch_analyze(self, file_paths: list) -> Dict[str, ForensicAnalysisResult]:
        """
        Analyze multiple documents.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary mapping file paths to analysis results

        Raises:
            Exception: If any analysis fails
        """
        results = {}

        for file_path in file_paths:
            result = self.analyze_document(file_path)
            results[file_path] = result
            logger.info(f"✓ Analyzed: {os.path.basename(file_path)} ({result.risk_level})")

        logger.info(f"Batch analysis complete: {len(results)} files processed")
        return results

    def get_supported_formats(self) -> list:
        """Get list of supported file formats."""
        return [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    def configure_thresholds(self, **kwargs) -> None:
        """
        Update analysis thresholds.

        Args:
            **kwargs: Threshold parameters to update

        Example:
            analyzer.configure_thresholds(
                ela_mean_threshold=3.0,
                copy_move_threshold=15
            )
        """
        self.config.update(**kwargs)
        logger.info(f"Updated thresholds: {list(kwargs.keys())}")

    def get_analysis_summary(self, result: ForensicAnalysisResult) -> Dict[str, Any]:
        """
        Get a summary of analysis findings.

        Args:
            result: ForensicAnalysisResult to summarize

        Returns:
            Dictionary with summary statistics
        """
        return result.get_summary_stats()

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return self.config.to_dict()

    def set_log_level(self, level: str) -> None:
        """
        Set logging level for forensic analysis.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        try:
            log_level = getattr(logging, level.upper())
            logging.getLogger("da3_obsidian.forensic_analysis").setLevel(log_level)
            logger.info(f"Log level set to {level.upper()}")
        except AttributeError as e:
            logger.error(f"Invalid log level: {level}")
            raise ValueError(f"Invalid log level: {level}") from e

    def verify_file_integrity(self, file_path: str, expected_hash: str) -> bool:
        """
        Verify file integrity using SHA256 hash.

        Args:
            file_path: Path to file to verify
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches, False otherwise
        """
        try:
            result = self.analyze_document(file_path, expected_hash)
            hash_info = result.raw_results.get("integrity_analysis", {}).get("file_hashes", {})
            return hash_info.get("matches_trusted_hash", False)
        except Exception as e:
            logger.error(f"File integrity verification failed: {e}")
            return False

    def get_risk_distribution(self, results: Dict[str, ForensicAnalysisResult]) -> Dict[str, int]:
        """
        Get distribution of risk levels from batch analysis results.

        Args:
            results: Dictionary of analysis results

        Returns:
            Dictionary with count of each risk level
        """
        distribution = {"Low": 0, "Medium": 0, "High": 0, "Very High": 0}

        for result in results.values():
            distribution[result.risk_level] += 1

        return distribution

    def export_results(self, result: ForensicAnalysisResult, output_file: str, format: str = "json") -> None:
        """
        Export analysis results to file.

        Args:
            result: Analysis result to export
            output_file: Output file path
            format: Export format ('json' or 'txt')
        """
        import json

        try:
            if format.lower() == "json":
                with open(output_file, "w") as f:
                    json.dump(result.raw_results, f, indent=2, default=str)
            elif format.lower() == "txt":
                with open(output_file, "w") as f:
                    f.write("Forensic Analysis Report\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"File: {os.path.basename(result.file_path)}\n")
                    f.write(f"Risk Level: {result.risk_level}\n")
                    f.write(f"Risk Score: {result.risk_score}/100\n")
                    f.write(f"Total Indicators: {result.total_issues}\n\n")

                    if result.fraud_indicators:
                        f.write("Fraud Indicators:\n")
                        for i, indicator in enumerate(result.fraud_indicators, 1):
                            f.write(f"  {i}. {indicator}\n")
                    else:
                        f.write("No fraud indicators detected.\n")
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Results exported to {output_file}")

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise
