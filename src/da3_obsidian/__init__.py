"""
DA3 Obsidian: Text extraction and anonymization toolkit.

This package provides comprehensive tools for extracting text from documents
and anonymizing personally identifiable information (PII).
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("da3-obsidian")
except PackageNotFoundError:
    # Fallback for development/uninstalled package
    __version__ = "0.0.0-dev"
