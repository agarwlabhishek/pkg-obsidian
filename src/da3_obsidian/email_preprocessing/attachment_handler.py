"""
Attachment handling module for extracting and managing email attachments.
"""

import logging
import mimetypes
import os
from typing import List, Set, Tuple

from .data_models import AttachmentData

logger = logging.getLogger(__name__)


class AttachmentHandler:
    """Handles extraction and management of email attachments."""

    def __init__(self, max_filename_length: int = 200):
        """
        Initialize attachment handler.

        Args:
            max_filename_length: Maximum allowed filename length
        """
        self.max_filename_length = max_filename_length
        self._seen_attachments: Set[Tuple[str, int]] = set()

        logger.debug("AttachmentHandler initialized")

    def extract_attachments(self, attachments, output_dir: str) -> List[AttachmentData]:
        """
        Extract file attachments to the specified output directory.
        Note: This method only handles file attachments, not embedded emails.

        Args:
            attachments: Collection of attachment objects from extract_msg
            output_dir: Directory to save attachments to

        Returns:
            List of AttachmentData objects
        """
        # Clear seen attachments for new extraction
        self._seen_attachments.clear()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        attachment_data = []

        try:
            for idx, attachment in enumerate(attachments):
                try:
                    # Only process file attachments, skip embedded emails
                    if hasattr(attachment, "data") and isinstance(attachment.data, bytes):
                        att_data = self._process_file_attachment(attachment, output_dir, idx)
                        if att_data:
                            attachment_data.append(att_data)
                except Exception as e:
                    logger.error(f"Error processing attachment {idx}: {e}")
                    continue

            logger.info(f"Extracted {len(attachment_data)} file attachments to {output_dir}")
            return attachment_data

        except Exception as e:
            logger.error(f"Error extracting attachments: {e}")
            return attachment_data

    def _process_file_attachment(self, attachment, output_dir: str, index: int) -> AttachmentData:
        """
        Process a single file attachment.

        Args:
            attachment: Attachment object
            output_dir: Output directory
            index: Attachment index for fallback naming

        Returns:
            AttachmentData object
        """
        # Get attachment data
        data = attachment.data
        size = len(data)

        # Get filename
        filename = self._get_attachment_filename(attachment, index)
        filename = self._sanitize_filename(filename)

        # Check for duplicates
        duplicate_key = (filename, size)
        if duplicate_key in self._seen_attachments:
            logger.debug(f"Skipping duplicate attachment: {filename}")
            return None

        self._seen_attachments.add(duplicate_key)

        # Generate unique filename if needed
        file_path = self._get_unique_filepath(output_dir, filename)

        try:
            # Write attachment to disk
            with open(file_path, "wb") as f:
                f.write(data)

            # Determine content type
            content_type = self._get_content_type(filename)

            attachment_data = AttachmentData(
                filename=os.path.basename(file_path),
                size=size,
                path=file_path,
                content_type=content_type,
                is_embedded_email=False,
            )

            logger.debug(f"Saved attachment: {file_path}")
            return attachment_data

        except Exception as e:
            logger.error(f"Error saving attachment {filename}: {e}")
            return None

    def _get_attachment_filename(self, attachment, index: int) -> str:
        """
        Get filename for attachment with fallback.

        Args:
            attachment: Attachment object
            index: Attachment index for fallback

        Returns:
            Filename string
        """
        # Try different filename attributes
        filename = None

        for attr in ["longFilename", "shortFilename", "displayName"]:
            if hasattr(attachment, attr):
                filename = getattr(attachment, attr)
                if filename and filename.strip():
                    break

        # Fallback to generic name
        if not filename or not filename.strip():
            filename = f"attachment_{index + 1}"

        return str(filename).strip()

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe filesystem use.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Remove leading/trailing whitespace and dots
        filename = filename.strip(" .")

        # Ensure filename is not empty
        if not filename:
            filename = "unnamed_attachment"

        # Truncate if too long, preserving extension
        if len(filename) > self.max_filename_length:
            name, ext = os.path.splitext(filename)
            max_name_length = self.max_filename_length - len(ext)
            filename = name[:max_name_length] + ext

        return filename

    def _get_unique_filepath(self, output_dir: str, filename: str) -> str:
        """
        Generate unique filepath to avoid overwriting existing files.

        Args:
            output_dir: Output directory
            filename: Desired filename

        Returns:
            Unique filepath
        """
        base_path = os.path.join(output_dir, filename)

        # If file doesn't exist, use as-is
        if not os.path.exists(base_path):
            return base_path

        # Generate unique name with counter
        name, ext = os.path.splitext(filename)
        counter = 1

        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_path = os.path.join(output_dir, new_filename)

            if not os.path.exists(new_path):
                return new_path

            counter += 1

            # Safety check to prevent infinite loop
            if counter > 1000:
                logger.warning(f"Too many duplicate files for: {filename}")
                return new_path

    def _get_content_type(self, filename: str) -> str:
        """
        Determine content type from filename.

        Args:
            filename: Filename to analyze

        Returns:
            MIME content type string
        """
        try:
            content_type, _ = mimetypes.guess_type(filename)
            return content_type or "application/octet-stream"
        except Exception:
            return "application/octet-stream"

    def get_attachment_stats(self, attachments: List[AttachmentData]) -> dict:
        """
        Get statistics about attachments.

        Args:
            attachments: List of AttachmentData objects

        Returns:
            Dictionary with attachment statistics
        """
        total_size = sum(att.size for att in attachments if not att.is_embedded_email)
        file_count = sum(1 for att in attachments if not att.is_embedded_email)
        embedded_count = sum(1 for att in attachments if att.is_embedded_email)

        # Count by content type
        content_types = {}
        for att in attachments:
            if not att.is_embedded_email and att.content_type:
                content_types[att.content_type] = content_types.get(att.content_type, 0) + 1

        return {
            "total_attachments": len(attachments),
            "file_attachments": file_count,
            "embedded_emails": embedded_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "content_types": content_types,
        }
