"""
Main email preprocessing interface for extracting data from .msg files.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, List, Union

from .data_models import EmailData
from .msg_processor import MSGProcessor

logger = logging.getLogger(__name__)


class EmailPreprocessor:
    """
    Main interface for email preprocessing and extraction.

    Provides comprehensive email data extraction from .msg files including:
    - Email metadata and content
    - Attachment extraction and management
    - Temporary file handling
    """

    def __init__(self, preserve_temp_files: bool = False):
        """
        Initialize email preprocessor.

        Args:
            preserve_temp_files: Whether to preserve temporary files after processing
        """
        self.preserve_temp_files = preserve_temp_files
        self.msg_processor = MSGProcessor()

        logger.info(f"EmailPreprocessor initialized (preserve_temp: {preserve_temp_files})")

    def process_msg_file(self, file_path: Union[str, Path]) -> EmailData:
        """
        Process a .msg file from disk.

        Args:
            file_path: Path to the .msg file

        Returns:
            EmailData containing extracted email information

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid .msg file
            Exception: For processing errors
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"MSG file not found: {file_path}")

        if not file_path.suffix.lower() == ".msg":
            raise ValueError(f"File is not a .msg file: {file_path}")

        try:
            logger.info(f"Processing MSG file: {file_path}")

            with open(file_path, "rb") as file_buffer:
                return self.process_msg_buffer(file_buffer, file_path.name)

        except Exception as e:
            logger.error(f"Failed to process MSG file '{file_path}': {e}")
            raise

    def process_msg_buffer(self, file_buffer: BinaryIO, filename: str) -> EmailData:
        """
        Process a .msg file from a buffer or file-like object.

        Args:
            file_buffer: File-like object containing .msg data
            filename: Original filename for reference

        Returns:
            EmailData containing extracted email information

        Raises:
            ValueError: If the buffer contains invalid data
            Exception: For processing errors
        """
        if not filename:
            filename = "unknown.msg"

        try:
            logger.info(f"Processing MSG buffer: {filename}")

            # Create temporary directory for attachments
            temp_dir = self._create_temp_directory()

            # Extract email data using MSG processor
            email_data = self.msg_processor.extract_from_buffer(file_buffer, filename, temp_dir)

            # Set temporary directory reference
            email_data.temp_directory = temp_dir

            logger.info(f"Successfully processed MSG: {filename}")
            logger.debug(f"Extracted {len(email_data.attachments)} attachments")

            return email_data

        except Exception as e:
            logger.error(f"Failed to process MSG buffer '{filename}': {e}")
            # Clean up temp directory on error
            if "temp_dir" in locals():
                self._cleanup_temp_directory(temp_dir)
            raise

    def cleanup_email_data(self, email_data: EmailData) -> None:
        """
        Clean up temporary files associated with email data.

        Args:
            email_data: EmailData object to clean up
        """
        if email_data.temp_directory and os.path.exists(email_data.temp_directory):
            try:
                self._cleanup_temp_directory(email_data.temp_directory)
                email_data.temp_directory = None
                logger.info("Cleaned up temporary email files")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")

    def get_attachment_count(self, email_data: EmailData) -> int:
        """
        Get total count of attachments.

        Args:
            email_data: EmailData to count attachments for

        Returns:
            Total number of attachments
        """
        return len(email_data.attachments)

    def list_all_files(self, email_data: EmailData) -> List[str]:
        """
        List all files extracted from the email.

        Args:
            email_data: EmailData to list files for

        Returns:
            List of relative file paths
        """
        files = []

        if email_data.temp_directory and os.path.exists(email_data.temp_directory):
            for root, _, filenames in os.walk(email_data.temp_directory):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, email_data.temp_directory)
                    files.append(rel_path)

        return files

    def get_email_summary(self, email_data: EmailData) -> dict:
        """
        Get summary statistics for email data.

        Args:
            email_data: EmailData to summarize

        Returns:
            Dictionary with summary information
        """
        total_attachments = self.get_attachment_count(email_data)
        embedded_count = sum(1 for att in email_data.attachments if att.is_embedded_email)

        body_length = len(email_data.body) if email_data.body else 0

        return {
            "filename": email_data.filename,
            "subject": email_data.subject,
            "sender": email_data.sender,
            "date": email_data.date,
            "total_attachments": total_attachments,
            "direct_attachments": len(email_data.attachments),
            "embedded_emails": embedded_count,
            "body_length": body_length,
            "has_temp_files": email_data.temp_directory is not None,
            "temp_directory": email_data.temp_directory,
        }

    def _create_temp_directory(self) -> str:
        """Create a temporary directory for email attachments."""
        temp_dir = os.path.join(tempfile.gettempdir(), "da3_msg_attachments")

        # Remove existing directory if it exists
        if os.path.exists(temp_dir):
            logger.debug(f"Removing existing temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        # Create new directory
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temp directory: {temp_dir}")

        return temp_dir

    def _cleanup_temp_directory(self, temp_dir: str) -> None:
        """Clean up temporary directory if not preserving files."""
        if not self.preserve_temp_files and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

    def __del__(self):
        """Cleanup on object destruction."""
        # Note: In production, explicit cleanup is preferred over relying on __del__
        pass
