"""
Text extraction module for DA3 Obsidian.

This module provides comprehensive document analysis and text extraction capabilities
using Azure Document Intelligence with automatic model selection and multi-format support.
"""

from .engines import (
    get_document_client,
    get_supported_models,
    initialize_client,
    is_client_initialized,
    validate_credentials,
)
from .extractor import DocumentExtractor, DocumentFeatures, DocumentPage, DocumentTable, ExtractionResult
from .models import MODEL_TYPES, SUPPORTED_FILE_EXTENSIONS

__all__ = [
    "DocumentExtractor",
    "ExtractionResult",
    "DocumentFeatures",
    "DocumentPage",
    "DocumentTable",
    "get_document_client",
    "get_supported_models",
    "initialize_client",
    "validate_credentials",
    "is_client_initialized",
    "MODEL_TYPES",
    "SUPPORTED_FILE_EXTENSIONS",
]
