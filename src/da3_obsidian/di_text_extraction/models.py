"""
Document Intelligence model definitions and constants.

This module defines the supported document types, model mappings, and file extensions
for the text extraction package.
"""

from typing import Any, Dict, List

# Supported file extensions for document processing
SUPPORTED_FILE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".pdf", ".tiff", ".tif", ".heif"]

# Document Intelligence model types mapping
MODEL_TYPES = {
    "general": "prebuilt-document",
    "invoice": "prebuilt-invoice",
    "receipt": "prebuilt-receipt",
    "id": "prebuilt-idDocument",
    "tax": "prebuilt-tax.us.w2",
    "layout": "prebuilt-layout",
    "business_card": "prebuilt-businessCard",
    "read": "prebuilt-read",
}

# Document type indicators with normalized weights for model detection
DOCUMENT_TYPE_INDICATORS = {
    "invoice": {
        "invoice": 15,
        "bill to": 10,
        "invoice no": 15,
        "invoice date": 10,
        "due date": 10,
        "payment terms": 10,
        "subtotal": 8,
        "total amount": 8,
        "amount due": 10,
        "tax id": 5,
        "purchase order": 5,
        "account number": 8,
        "po number": 8,
    },
    "receipt": {
        "receipt": 15,
        "customer copy": 10,
        "merchant": 8,
        "total": 8,
        "change": 10,
        "cash": 5,
        "payment method": 10,
        "item": 5,
        "qty": 5,
        "thank you for your purchase": 10,
        "cashier": 8,
        "transaction": 8,
        "return policy": 8,
    },
    "id": {
        "identification": 10,
        "identity card": 15,
        "date of birth": 15,
        "expiration date": 15,
        "license": 10,
        "passport": 15,
        "id number": 15,
        "document number": 15,
        "nationality": 10,
        "gender": 5,
        "signature": 5,
        "issued by": 10,
        "driver": 8,
    },
    "business_card": {
        "tel": 8,
        "phone": 8,
        "email": 10,
        "www": 8,
        "website": 8,
        "fax": 5,
        "mobile": 8,
        "position": 10,
        "title": 8,
        "company": 10,
        "address": 8,
        "linkedin": 8,
        "ceo": 8,
        "director": 8,
    },
    "tax": {
        "w-2": 20,
        "w2": 20,
        "wage": 8,
        "tax statement": 15,
        "earnings summary": 10,
        "federal income tax": 15,
        "social security tax": 15,
        "medicare tax": 15,
        "employer identification number": 15,
        "tax year": 10,
        "1099": 20,
        "tax form": 15,
        "irs": 15,
        "filing status": 10,
    },
}

# Model detection configuration
MODEL_DETECTION_CONFIG = {
    "base_confidence_threshold": 60,
    "min_confidence_gap": 15,
    "disambiguation_factor": 0.7,
    "file_size_thresholds": {
        "small": 50000,  # Very small document (likely business card)
        "medium": 100000,  # Small document (likely receipt/ID)
        "large": 500000,  # Large document (likely invoice/tax)
    },
}


def get_model_id(model_type: str) -> str:
    """
    Get the Azure Document Intelligence model ID for a given model type.

    Args:
        model_type: The model type key

    Returns:
        Azure Document Intelligence model ID

    Raises:
        ValueError: If model type is not supported
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type '{model_type}'. " f"Supported types: {', '.join(MODEL_TYPES.keys())}")

    return MODEL_TYPES[model_type]


def get_supported_model_types() -> List[str]:
    """
    Get list of supported model types.

    Returns:
        List of model type keys
    """
    return list(MODEL_TYPES.keys())


def get_content_type(file_extension: str) -> str:
    """
    Get appropriate content type for file extension.

    Args:
        file_extension: File extension (with or without dot)

    Returns:
        Content type string for Azure Document Intelligence
    """
    # Normalize extension
    if not file_extension.startswith("."):
        file_extension = f".{file_extension}"

    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        return "application/pdf"
    elif file_extension in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif file_extension == ".png":
        return "image/png"
    elif file_extension == ".bmp":
        return "image/bmp"
    elif file_extension in [".tiff", ".tif"]:
        return "image/tiff"
    elif file_extension == ".heif":
        return "image/heif"
    else:
        # Default to generic image type for unknown extensions
        return f"image/{file_extension[1:]}"


def validate_file_extension(file_extension: str) -> bool:
    """
    Validate if file extension is supported.

    Args:
        file_extension: File extension to validate

    Returns:
        True if extension is supported
    """
    # Normalize extension
    if not file_extension.startswith("."):
        file_extension = f".{file_extension}"

    return file_extension.lower() in SUPPORTED_FILE_EXTENSIONS


def get_document_type_indicators(document_type: str) -> Dict[str, int]:
    """
    Get document type indicators for model detection.

    Args:
        document_type: Document type to get indicators for

    Returns:
        Dictionary of indicators and their weights

    Raises:
        ValueError: If document type is not supported
    """
    if document_type not in DOCUMENT_TYPE_INDICATORS:
        raise ValueError(
            f"Unsupported document type '{document_type}'. "
            f"Supported types: {', '.join(DOCUMENT_TYPE_INDICATORS.keys())}"
        )

    return DOCUMENT_TYPE_INDICATORS[document_type].copy()


def get_detection_config() -> Dict[str, Any]:
    """
    Get model detection configuration.

    Returns:
        Dictionary with detection parameters
    """
    return MODEL_DETECTION_CONFIG.copy()
