"""
Azure Document Intelligence engine management for text extraction.

This module handles the initialization and management of Azure Document Intelligence
client with proper credential validation and configuration.
"""

import logging
import os
from typing import Dict, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError

from .models import MODEL_TYPES

logger = logging.getLogger(__name__)

# Global client instance
_document_client: Optional[DocumentIntelligenceClient] = None
_initialized = False


class DocumentIntelligenceEngineManager:
    """Manages Azure Document Intelligence client with singleton pattern."""

    def __init__(self):
        """Initialize engine manager."""
        self._client: Optional[DocumentIntelligenceClient] = None
        self._endpoint: Optional[str] = None
        self._initialized = False

    def initialize_client(self, endpoint: Optional[str] = None, key: Optional[str] = None) -> None:
        """
        Initialize Azure Document Intelligence client.

        Args:
            endpoint: Azure Document Intelligence endpoint (uses env var if None)
            key: Azure Document Intelligence API key (uses env var if None)

        Raises:
            KeyError: If required environment variables are missing
            ClientAuthenticationError: If authentication fails
        """
        global _document_client, _initialized

        # Use provided credentials or fail fast on missing environment variables
        if endpoint is None:
            endpoint = os.environ["AZURE_DI_ENDPOINT"]
        if key is None:
            key = os.environ["AZURE_DI_KEY"]

        try:
            logger.debug(f"Creating Document Intelligence client with endpoint: {endpoint[:10]}...")

            credential = AzureKeyCredential(key)
            _document_client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)

            self._client = _document_client
            self._endpoint = endpoint
            self._initialized = True
            _initialized = True

            logger.info("Document Intelligence client initialized successfully")

        except ClientAuthenticationError:
            logger.error("Authentication failed")
            raise
        except Exception as e:
            logger.error(f"Failed to create Document Intelligence client: {str(e)}")
            raise

    def get_client(self) -> DocumentIntelligenceClient:
        """
        Get the Document Intelligence client instance.

        Returns:
            DocumentIntelligenceClient instance

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._initialized or self._client is None:
            # Try to initialize with environment variables
            self.initialize_client()

        return self._client

    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized and self._client is not None

    def get_supported_models(self) -> Dict[str, str]:
        """Get supported model types."""
        return MODEL_TYPES.copy()

    def validate_credentials(self, endpoint: Optional[str] = None, key: Optional[str] = None) -> bool:
        """
        Validate Azure Document Intelligence credentials.

        Args:
            endpoint: Endpoint to validate (uses env var if None)
            key: API key to validate (uses env var if None)

        Returns:
            True if credentials are valid

        Raises:
            KeyError: If required environment variables are missing
            ClientAuthenticationError: If credentials are invalid
        """
        if endpoint is None:
            endpoint = os.environ["AZURE_DI_ENDPOINT"]
        if key is None:
            key = os.environ["AZURE_DI_KEY"]

        try:
            # Create temporary client for validation
            credential = AzureKeyCredential(key)
            DocumentIntelligenceClient(endpoint=endpoint, credential=credential)

            logger.info("Credentials validated successfully")
            return True

        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            raise

    def reset_client(self) -> None:
        """Reset client (mainly for testing purposes)."""
        global _document_client, _initialized

        self._client = None
        self._endpoint = None
        self._initialized = False
        _document_client = None
        _initialized = False

        logger.info("Document Intelligence client reset")


# Global manager instance
_engine_manager = DocumentIntelligenceEngineManager()


def get_document_client(endpoint: Optional[str] = None, key: Optional[str] = None) -> DocumentIntelligenceClient:
    """
    Get the Document Intelligence client instance.

    Args:
        endpoint: Azure Document Intelligence endpoint (uses env var if None)
        key: Azure Document Intelligence API key (uses env var if None)

    Returns:
        DocumentIntelligenceClient instance

    Raises:
        KeyError: If required environment variables are missing
        ClientAuthenticationError: If authentication fails
    """
    if not _engine_manager.is_initialized():
        _engine_manager.initialize_client(endpoint, key)

    return _engine_manager.get_client()


def get_supported_models() -> Dict[str, str]:
    """
    Get supported Document Intelligence model types.

    Returns:
        Dictionary mapping model names to model IDs
    """
    return _engine_manager.get_supported_models()


def initialize_client(endpoint: Optional[str] = None, key: Optional[str] = None) -> None:
    """
    Initialize Document Intelligence client with custom configuration.

    Args:
        endpoint: Azure Document Intelligence endpoint
        key: Azure Document Intelligence API key

    Raises:
        KeyError: If required environment variables are missing
        ClientAuthenticationError: If authentication fails
    """
    _engine_manager.initialize_client(endpoint, key)


def validate_credentials(endpoint: Optional[str] = None, key: Optional[str] = None) -> bool:
    """
    Validate Azure Document Intelligence credentials.

    Args:
        endpoint: Endpoint to validate (uses env var if None)
        key: API key to validate (uses env var if None)

    Returns:
        True if credentials are valid

    Raises:
        KeyError: If required environment variables are missing
        ClientAuthenticationError: If credentials are invalid
    """
    return _engine_manager.validate_credentials(endpoint, key)


def is_client_initialized() -> bool:
    """
    Check if Document Intelligence client is initialized.

    Returns:
        True if client is ready for use
    """
    return _engine_manager.is_initialized()


def reset_client() -> None:
    """Reset client (mainly for testing)."""
    _engine_manager.reset_client()
