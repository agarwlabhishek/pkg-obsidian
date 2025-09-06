"""
MSG file processing module for extracting email data and attachments.
"""

import logging
from io import BytesIO
from typing import BinaryIO

try:
    import extract_msg
except ImportError:
    extract_msg = None

from .attachment_handler import AttachmentHandler
from .data_models import EmailData

logger = logging.getLogger(__name__)


class MSGProcessor:
    """Handles processing of .msg files using the extract_msg library."""

    def __init__(self):
        """Initialize MSG processor."""
        if extract_msg is None:
            raise ImportError(
                "extract_msg library is required for MSG processing. " "Install it with: pip install extract-msg"
            )

        self.attachment_handler = AttachmentHandler()
        logger.debug("MSGProcessor initialized")

    def extract_from_buffer(self, file_buffer: BinaryIO, filename: str, output_dir: str) -> EmailData:
        """
        Extract email data from a .msg file buffer.

        Args:
            file_buffer: File-like object containing .msg data
            filename: Original filename for reference
            output_dir: Directory to extract attachments to

        Returns:
            EmailData containing extracted information

        Raises:
            ValueError: If the buffer contains invalid .msg data
            Exception: For extraction errors
        """
        try:
            # Ensure we have a BytesIO object
            if not isinstance(file_buffer, BytesIO):
                if hasattr(file_buffer, "read"):
                    content = file_buffer.read()
                    if hasattr(file_buffer, "seek"):
                        file_buffer.seek(0)  # Reset position for potential reuse
                    file_buffer = BytesIO(content)
                else:
                    raise ValueError("Invalid file buffer provided")

            # Extract message using extract_msg
            msg = extract_msg.Message(file_buffer)

            # Extract basic email information
            email_data = self._extract_email_metadata(msg, filename)

            # Extract file attachments (skip embedded emails for now)
            email_data.attachments = self.attachment_handler.extract_attachments(msg.attachments, output_dir)

            # TODO: Add embedded email processing in future version
            # For now, we'll just note if there are any embedded emails
            embedded_count = 0
            for attachment in msg.attachments:
                if hasattr(attachment, "data") and not isinstance(attachment.data, bytes):
                    embedded_count += 1

            if embedded_count > 0:
                logger.info(f"Note: {embedded_count} embedded emails found but not processed in this version")

            logger.info(f"Successfully extracted MSG data for: {filename}")
            return email_data

        except Exception as e:
            logger.error(f"Error extracting MSG file '{filename}': {e}")
            raise

    def _extract_email_metadata(self, msg, filename: str) -> EmailData:
        """
        Extract basic email metadata from message object.

        Args:
            msg: extract_msg Message object
            filename: Original filename

        Returns:
            EmailData with basic information
        """
        try:
            # Safely extract attributes with fallbacks
            subject = self._safe_getattr(msg, "subject")
            body = self._safe_getattr(msg, "body")
            sender = self._safe_getattr(msg, "sender")
            to = self._safe_getattr(msg, "to")
            cc = self._safe_getattr(msg, "cc")
            bcc = self._safe_getattr(msg, "bcc")
            date = self._safe_getattr(msg, "date")
            message_id = self._safe_getattr(msg, "messageId")
            importance = self._safe_getattr(msg, "importance")

            # Convert date to string if it exists
            if date is not None:
                date = str(date)

            email_data = EmailData(
                filename=filename,
                subject=subject,
                body=body,
                sender=sender,
                to=to,
                cc=cc,
                bcc=bcc,
                date=date,
                message_id=message_id,
                importance=importance,
                attachments=[],  # Will be populated by attachment handler
            )

            logger.debug(f"Extracted metadata for: {filename}")
            return email_data

        except Exception as e:
            logger.error(f"Error extracting email metadata: {e}")
            # Return minimal EmailData on error
            return EmailData(
                filename=filename,
                subject=None,
                body=None,
                sender=None,
                to=None,
                cc=None,
                bcc=None,
                date=None,
                attachments=[],
            )

    def _safe_getattr(self, obj, attr: str, default=None):
        """
        Safely get attribute from object with fallback.

        Args:
            obj: Object to get attribute from
            attr: Attribute name
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        try:
            value = getattr(obj, attr, default)
            # Handle empty strings and None values
            if value == "" or value is None:
                return default
            return value
        except Exception as e:
            logger.debug(f"Error getting attribute '{attr}': {e}")
            return default

    @staticmethod
    def is_msg_file(file_path: str) -> bool:
        """
        Check if a file is a .msg file based on extension.

        Args:
            file_path: Path to check

        Returns:
            True if file appears to be a .msg file
        """
        return file_path.lower().endswith(".msg")

    @staticmethod
    def validate_msg_buffer(buffer: BinaryIO) -> bool:
        """
        Validate if a buffer contains valid .msg data.

        Args:
            buffer: Buffer to validate

        Returns:
            True if buffer appears to contain valid .msg data
        """
        try:
            # Save current position
            original_pos = buffer.tell() if hasattr(buffer, "tell") else 0

            # Try to read first few bytes to check for MSG signature
            if hasattr(buffer, "seek"):
                buffer.seek(0)

            header = buffer.read(8)

            # Reset position
            if hasattr(buffer, "seek"):
                buffer.seek(original_pos)

            # Check for OLE/COM document signature (MSG files are OLE documents)
            return header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")

        except Exception:
            return False
