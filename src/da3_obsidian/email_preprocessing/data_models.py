"""
Data models for email preprocessing module.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AttachmentData:
    """Represents an email attachment."""

    filename: str
    size: int
    path: str
    content_type: Optional[str] = None
    is_embedded_email: bool = False
    embedded_email_data: Optional["EmailData"] = None


@dataclass
class EmailData:
    """Represents extracted email data."""

    filename: str
    subject: Optional[str]
    body: Optional[str]
    sender: Optional[str]
    to: Optional[str]
    cc: Optional[str]
    bcc: Optional[str]
    date: Optional[str]
    attachments: List[AttachmentData]
    temp_directory: Optional[str] = None
    message_id: Optional[str] = None
    importance: Optional[str] = None
