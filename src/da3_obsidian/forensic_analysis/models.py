"""
Data models for forensic analysis results using Pydantic.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class AnalysisType(str, Enum):
    """Types of forensic analysis."""

    METADATA = "metadata"
    IMAGE_MANIPULATION = "image_manipulation"
    STRUCTURE = "structure"
    TEXT_CONSISTENCY = "text_consistency"


class FileInfo(BaseModel):
    """Basic file information."""

    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    file_size_human: str
    file_modified: str
    file_created: str
    analysis_timestamp: str


class HashInfo(BaseModel):
    """File hash information."""

    md5: str
    sha256: str
    matches_trusted_hash: Optional[bool] = None


class MetadataIssue(BaseModel):
    """Represents a metadata inconsistency."""

    issue_type: str
    description: str
    severity: str
    field_name: Optional[str] = None


class ImageAnalysisResult(BaseModel):
    """Result of image forensic analysis."""

    ela_suspicious: bool = False
    copy_move_detected: bool = False
    noise_inconsistent: bool = False
    edge_artifacts: bool = False
    jpeg_ghosts: bool = False
    shadow_inconsistent: bool = False
    cfa_suspicious: bool = False

    def has_manipulation_indicators(self) -> bool:
        """Check if any manipulation indicators are present."""
        return any(
            [
                self.ela_suspicious,
                self.copy_move_detected,
                self.noise_inconsistent,
                self.edge_artifacts,
                self.jpeg_ghosts,
                self.shadow_inconsistent,
                self.cfa_suspicious,
            ]
        )


class PDFStructureResult(BaseModel):
    """Result of PDF structure analysis."""

    version_mismatch: bool = False
    has_javascript: bool = False
    is_encrypted: bool = False
    has_signatures: bool = False
    incremental_updates: int = 0
    embedded_files: int = 0
    has_layers: bool = False
    uncommon_filters: List[str] = Field(default_factory=list)

    def has_structure_anomalies(self) -> bool:
        """Check if any structure anomalies are present."""
        return any(
            [
                self.version_mismatch,
                self.has_javascript,
                self.incremental_updates > 0,
                self.embedded_files > 0,
                self.has_layers,
                len(self.uncommon_filters) > 0,
            ]
        )


class TextAnalysisResult(BaseModel):
    """Result of text consistency analysis."""

    excessive_fonts: bool = False
    font_count: int = 0
    pages_with_text: int = 0
    total_text_length: int = 0

    def has_text_issues(self) -> bool:
        """Check if any text inconsistencies are present."""
        return self.excessive_fonts


class AnalysisResult(BaseModel):
    """Comprehensive analysis result."""

    # File information
    file_info: FileInfo
    hash_info: HashInfo

    # Risk assessment
    risk_score: int
    risk_level: RiskLevel
    fraud_indicators: List[str]

    # Analysis results
    metadata_issues: List[str]
    image_analysis: Optional[ImageAnalysisResult] = None
    pdf_structure: Optional[PDFStructureResult] = None
    text_analysis: Optional[TextAnalysisResult] = None

    # Analysis status
    analysis_successful: bool = True
    analysis_error: Optional[str] = None
    analysis_completed: str = ""

    # Raw data
    raw_results: Optional[Dict[str, Any]] = None

    @property
    def has_metadata_issues(self) -> bool:
        """Check if metadata issues are present."""
        return len(self.metadata_issues) > 0

    @property
    def has_image_manipulation(self) -> bool:
        """Check if image manipulation indicators are present."""
        return self.image_analysis and self.image_analysis.has_manipulation_indicators()

    @property
    def has_structure_anomalies(self) -> bool:
        """Check if PDF structure anomalies are present."""
        return self.pdf_structure and self.pdf_structure.has_structure_anomalies()

    @property
    def has_text_inconsistencies(self) -> bool:
        """Check if text inconsistencies are present."""
        return self.text_analysis and self.text_analysis.has_text_issues()

    @property
    def total_issues(self) -> int:
        """Get total number of fraud indicators."""
        return len(self.fraud_indicators)

    @property
    def is_high_risk(self) -> bool:
        """Check if result indicates high risk."""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]

    def get_issues_by_category(self) -> Dict[str, List[str]]:
        """Categorize fraud indicators by analysis type."""
        categories = {"metadata": [], "image_manipulation": [], "structure_anomalies": [], "text_inconsistencies": []}

        for indicator in self.fraud_indicators:
            indicator_lower = indicator.lower()

            if "metadata" in indicator_lower:
                categories["metadata"].append(indicator)
            elif any(word in indicator_lower for word in ["image", "manipulation", "ela", "ghost", "copy-move"]):
                categories["image_manipulation"].append(indicator)
            elif any(word in indicator_lower for word in ["structure", "javascript", "layer", "update", "signature"]):
                categories["structure_anomalies"].append(indicator)
            elif any(word in indicator_lower for word in ["font", "text"]):
                categories["text_inconsistencies"].append(indicator)
            else:
                categories["metadata"].append(indicator)

        return categories

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        issues_by_category = self.get_issues_by_category()

        return {
            "file_name": self.file_info.file_name,
            "file_size": self.file_info.file_size_human,
            "risk_assessment": {
                "score": self.risk_score,
                "level": self.risk_level.value,
                "is_high_risk": self.is_high_risk,
            },
            "findings": {
                "total_indicators": self.total_issues,
                "by_category": {k: len(v) for k, v in issues_by_category.items()},
                "has_issues": {
                    "metadata": self.has_metadata_issues,
                    "images": self.has_image_manipulation,
                    "structure": self.has_structure_anomalies,
                    "text": self.has_text_inconsistencies,
                },
            },
            "analysis_status": {
                "successful": self.analysis_successful,
                "completed": self.analysis_completed,
                "error": self.analysis_error,
            },
        }

    @classmethod
    def from_raw_results(cls, raw_results: Dict[str, Any]) -> "AnalysisResult":
        """Create AnalysisResult from raw analysis dictionary."""
        file_info_data = raw_results.get("file_info", {})
        file_info = FileInfo(
            file_path=file_info_data.get("file_path", ""),
            file_name=file_info_data.get("file_name", ""),
            file_extension=file_info_data.get("file_extension", ""),
            file_size_bytes=file_info_data.get("file_size_bytes", 0),
            file_size_human=file_info_data.get("file_size_human", ""),
            file_modified=file_info_data.get("file_modified", ""),
            file_created=file_info_data.get("file_created", ""),
            analysis_timestamp=file_info_data.get("analysis_timestamp", ""),
        )

        hash_info_data = raw_results.get("integrity_analysis", {}).get("file_hashes", {})
        hash_info = HashInfo(
            md5=hash_info_data.get("md5", ""),
            sha256=hash_info_data.get("sha256", ""),
            matches_trusted_hash=hash_info_data.get("matches_trusted_hash"),
        )

        summary = raw_results.get("summary", {})
        fraud_indicators = summary.get("potential_fraud_indicators", [])
        risk_score = summary.get("risk_score", 0)
        risk_level_str = summary.get("risk_level", "Low")

        risk_level = RiskLevel.LOW
        for level in RiskLevel:
            if level.value == risk_level_str:
                risk_level = level
                break

        metadata_analysis = raw_results.get("metadata_analysis", {})
        metadata_issues = metadata_analysis.get("metadata_issues", [])

        image_analysis = None
        image_data = raw_results.get("image_analysis", {})
        if image_data:
            image_analysis = ImageAnalysisResult(
                ela_suspicious=image_data.get("error_level_analysis", {}).get("suspicious", False),
                copy_move_detected=image_data.get("copy_move_detection", {}).get("suspicious", False),
                noise_inconsistent=image_data.get("noise_analysis", {}).get("inconsistent_noise", False),
                edge_artifacts=image_data.get("edge_integrity", {}).get("suspicious", False),
                jpeg_ghosts=image_data.get("jpeg_ghost_analysis", {}).get("ghosts_detected", False),
                shadow_inconsistent=image_data.get("shadow_consistency", {}).get("suspicious", False),
                cfa_suspicious=image_data.get("cfa_interpolation", {}).get("suspicious", False),
            )

        pdf_structure = None
        structure_data = raw_results.get("integrity_analysis", {}).get("pdf_structure", {})
        if structure_data:
            pdf_structure = PDFStructureResult(
                version_mismatch="version_mismatch" in structure_data,
                has_javascript=structure_data.get("has_javascript", False),
                is_encrypted=structure_data.get("is_encrypted", False),
                has_signatures=structure_data.get("has_signatures", False),
                incremental_updates=structure_data.get("incremental_updates", 0),
                embedded_files=len(structure_data.get("embedded_files", [])),
                has_layers=structure_data.get("has_layers", False),
                uncommon_filters=structure_data.get("filters_used", []),
            )

        text_analysis = None
        text_data = raw_results.get("text_analysis", {})
        if text_data:
            font_analysis = text_data.get("font_analysis", {})
            text_extraction = text_data.get("text_extraction", {})

            text_analysis = TextAnalysisResult(
                excessive_fonts=font_analysis.get("distinct_fonts", 0) > 3,
                font_count=font_analysis.get("distinct_fonts", 0),
                pages_with_text=text_extraction.get("pages_with_text", 0),
                total_text_length=text_extraction.get("total_text_length", 0),
            )

        return cls(
            file_info=file_info,
            hash_info=hash_info,
            risk_score=risk_score,
            risk_level=risk_level,
            fraud_indicators=fraud_indicators,
            metadata_issues=metadata_issues,
            image_analysis=image_analysis,
            pdf_structure=pdf_structure,
            text_analysis=text_analysis,
            analysis_successful=summary.get("analysis_successful", True),
            analysis_error=summary.get("analysis_error", {}).get("message") if "analysis_error" in summary else None,
            analysis_completed=summary.get("analysis_completed", ""),
            raw_results=raw_results,
        )
