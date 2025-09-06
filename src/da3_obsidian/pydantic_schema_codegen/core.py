"""
Core functionality for schema normalization, validation, and dynamic class creation.
"""

from __future__ import annotations
import importlib
import logging
from typing import Any, Dict, Tuple, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _looks_like_class_def(obj: Any) -> bool:
    """Check if an object looks like a class definition dictionary."""
    return isinstance(obj, dict) and "fields" in obj and isinstance(obj["fields"], dict)


def normalize_schema_payload(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Normalize schema payload to standard format.

    Returns (normalized, wrapped) where normalized always has 'classes'.

    - If raw already has 'classes' (non-empty dict), return unchanged.
    - If raw is a bare mapping whose values resemble ClassDefinition, wrap as {'classes': raw}.
    - Otherwise, raise ValueError with actionable message.

    Args:
        raw: Raw schema dictionary

    Returns:
        Tuple of (normalized_schema, was_wrapped)

    Raises:
        ValueError: If schema format is unsupported
    """
    if "classes" in raw:
        classes = raw["classes"]
        if not isinstance(classes, dict) or not classes:
            raise ValueError("'classes' must be a non-empty dict")
        return raw, False

    if isinstance(raw, dict) and raw:
        # Check if this might be a bare mapping of class definitions
        # Look for a value that's not a reserved schema field
        sample_key = next((k for k in raw.keys() if k not in ["executive_summary", "assumptions"]), None)

        if sample_key and _looks_like_class_def(raw[sample_key]):
            # It's a bare mapping of classes, wrap it
            wrapped = {"classes": raw}
            logger.debug("Normalized bare mapping payload to wrapped form with 'classes'.")
            return wrapped, True

    raise ValueError("Unsupported schema payload: expected key 'classes' or a mapping of class names → definitions.")


def get_schema_class(path: str) -> Type[BaseModel]:
    """
    Import schema validation class from module path.

    Args:
        path: Module path in format "module_path:ClassName"

    Returns:
        Schema validation class

    Raises:
        ValueError: If path format is invalid
        ImportError: If module or class cannot be imported
        AttributeError: If class doesn't exist in module
    """
    try:
        module_name, class_name = path.split(":", 1)
    except ValueError as e:
        raise ValueError(f"Schema class path must be of the form module:ClassName; got {path!r}") from e

    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        logger.debug(f"Successfully imported schema class: {path}")
        return cls
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}': {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}': {e}") from e


def validate_schema(raw: Dict[str, Any], schema_class: Type[BaseModel]) -> BaseModel:
    """
    Validate schema dictionary against Pydantic schema class.

    Args:
        raw: Raw schema dictionary to validate
        schema_class: Pydantic model class for validation

    Returns:
        Validated schema model instance

    Raises:
        ValidationError: If schema validation fails
    """
    try:
        validated = schema_class.model_validate(raw)  # Pydantic v2
        logger.debug("Schema validation successful")
        return validated
    except ValidationError as ve:
        logger.error(f"Schema validation failed: {ve}")
        raise


def build_dynamic_classes(resp: BaseModel) -> Dict[str, Type]:
    """
    Build dynamic Pydantic classes from validated schema response.

    Args:
        resp: Validated schema response model

    Returns:
        Dictionary mapping class names to dynamically created Pydantic classes

    Raises:
        AttributeError: If response model doesn't have to_pydantic_classes method
        RuntimeError: If dynamic class creation fails
    """
    try:
        if not hasattr(resp, "to_pydantic_classes"):
            raise AttributeError(f"Schema response model {type(resp).__name__} must have 'to_pydantic_classes' method")

        classes = resp.to_pydantic_classes()
        logger.info(f"Created {len(classes)} dynamic classes")
        logger.debug(f"Dynamic class names: {list(classes.keys())}")
        return classes
    except Exception as e:
        logger.error(f"Dynamic class creation failed: {e}")
        raise RuntimeError(f"Failed to create dynamic classes: {e}") from e


def get_class_info_from_schema(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract class information from normalized schema.

    Args:
        schema: Normalized schema dictionary

    Returns:
        Dictionary mapping class names to their definitions

    Raises:
        ValueError: If schema format is invalid
    """
    if "classes" not in schema:
        raise ValueError("Schema missing 'classes' key")

    classes = schema["classes"]
    if not isinstance(classes, dict):
        raise ValueError("'classes' must be a dictionary")

    return classes


def validate_class_definition(class_name: str, class_def: Dict[str, Any]) -> None:
    """
    Validate individual class definition structure.

    Args:
        class_name: Name of the class
        class_def: Class definition dictionary

    Raises:
        ValueError: If class definition is invalid
    """
    if not isinstance(class_def, dict):
        raise ValueError(f"Class '{class_name}' definition must be a dictionary")

    if "fields" not in class_def:
        raise ValueError(f"Class '{class_name}' missing 'fields' key")

    fields = class_def["fields"]
    if not isinstance(fields, dict):
        raise ValueError(f"Class '{class_name}' fields must be a dictionary")

    # Validate each field
    for field_name, field_spec in fields.items():
        if not isinstance(field_spec, dict):
            raise ValueError(f"Field '{field_name}' in class '{class_name}' must be a dictionary")
