"""
Email preprocessing module for DA3 Obsidian.

This module provides comprehensive email extraction and preprocessing capabilities
from .msg files with attachment handling.
"""

from .attachment_handler import AttachmentHandler
from .data_models import AttachmentData, EmailData
from .msg_processor import MSGProcessor
from .preprocessor import EmailPreprocessor

__all__ = [
    "EmailPreprocessor",
    "EmailData",
    "AttachmentData",
    "MSGProcessor",
    "AttachmentHandler",
]
