"""
Text anonymization module for DA3 Obsidian.

This module provides comprehensive PII detection and anonymization capabilities
using Microsoft Presidio with multi-language support.
"""

from .anonymizer import AnonymizationResult, PIIEntity, TextAnonymizer
from .engines import (
    get_analyzer_engine,
    get_anonymizer_engine,
    get_initialized_languages,
    get_supported_languages,
    initialize_engines,
    initialize_language,
    is_language_initialized,
    validate_language_support,
)

__all__ = [
    "TextAnonymizer",
    "PIIEntity",
    "AnonymizationResult",
    "get_analyzer_engine",
    "get_anonymizer_engine",
    "get_supported_languages",
    "get_initialized_languages",
    "is_language_initialized",
    "initialize_language",
    "initialize_engines",
    "validate_language_support",
]
