"""
Forensic analysis module for DA3 Obsidian.

This module provides comprehensive document fraud detection capabilities
including metadata analysis, image manipulation detection, and structure verification.
"""

# Import main components
from .analyzer import ForensicAnalysisResult, ForensicAnalyzer
from .config import ForensicConfig
from .core import DocumentAnalyzer
from .image_analyzer import ImageAnalyzer
from .metadata_analyzer import MetadataAnalyzer
from .models import (
    AnalysisResult,
    AnalysisType,
    FileInfo,
    HashInfo,
    ImageAnalysisResult,
    MetadataIssue,
    PDFStructureResult,
    RiskLevel,
    TextAnalysisResult,
)

# Import core components (available for advanced usage)
from .pdf_analyzer import PDFAnalyzer

__all__ = [
    # Main interface
    "ForensicAnalyzer",
    "ForensicAnalysisResult",
    "ForensicConfig",
    # Data models
    "AnalysisResult",
    "RiskLevel",
    "AnalysisType",
    "FileInfo",
    "HashInfo",
    "MetadataIssue",
    "ImageAnalysisResult",
    "PDFStructureResult",
    "TextAnalysisResult",
    # Core components (advanced usage)
    "DocumentAnalyzer",
    "PDFAnalyzer",
    "ImageAnalyzer",
    "MetadataAnalyzer",
]
