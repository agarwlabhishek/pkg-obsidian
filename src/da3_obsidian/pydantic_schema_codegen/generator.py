"""
Main schema code generation interface for converting structured schemas to Pydantic models.
Enhanced with source code parsing capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from .core import (
    build_dynamic_classes,
    normalize_schema_payload,
    validate_schema,
)
from .schemas import StructuredSchemaResponse
from .source_generator import generate_static_source
from .source_parser import parse_pydantic_source

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationResult:
    """Result of schema code generation operation."""

    generated_source: str
    classes_generated: int
    dynamic_classes: Optional[Dict[str, Type]] = None
    original_schema: Optional[Dict[str, Any]] = None
    normalized_schema: Optional[Dict[str, Any]] = None
    was_wrapped: bool = False


@dataclass
class ParseResult:
    """Result of source code parsing operation."""

    parsed_schema: Dict[str, Any]
    classes_found: int
    original_source: str
    normalized_schema: Optional[Dict[str, Any]] = None
    was_wrapped: bool = False


class SchemaCodeGenerator:
    """
    Main interface for Pydantic schema code generation.

    Provides schema normalization, validation, dynamic class creation,
    static source code generation, and source code parsing capabilities.
    """

    def __init__(self, schema_class: Type[BaseModel] = StructuredSchemaResponse):
        """
        Initialize schema code generator.

        Args:
            schema_class: Schema validation class (defaults to StructuredSchemaResponse)
        """
        self.schema_class = schema_class
        logger.info(f"SchemaCodeGenerator initialized with schema class: {schema_class.__name__}")

    def normalize_schema(self, raw_schema: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """
        Normalize schema payload to standard format.

        Args:
            raw_schema: Raw schema dictionary

        Returns:
            Tuple of (normalized_schema, was_wrapped)

        Raises:
            ValueError: If schema format is invalid
        """
        try:
            normalized, wrapped = normalize_schema_payload(raw_schema)
            logger.debug(f"Schema normalized, was_wrapped: {wrapped}")
            return normalized, wrapped
        except Exception as e:
            logger.error(f"Schema normalization failed: {e}")
            raise

    def validate_schema(self, normalized_schema: Dict[str, Any]) -> BaseModel:
        """
        Validate normalized schema against schema class.

        Args:
            normalized_schema: Normalized schema dictionary

        Returns:
            Validated schema model instance

        Raises:
            ValidationError: If schema validation fails
        """
        try:
            validated = validate_schema(normalized_schema, self.schema_class)
            logger.debug("Schema validation successful")
            return validated
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise

    def create_dynamic_classes(self, validated_schema: BaseModel) -> Dict[str, Type]:
        """
        Create dynamic Pydantic classes from validated schema.

        Args:
            validated_schema: Validated schema model

        Returns:
            Dictionary mapping class names to dynamically created classes

        Raises:
            RuntimeError: If dynamic class creation fails
        """
        try:
            dynamic_classes = build_dynamic_classes(validated_schema)
            logger.info(f"Created {len(dynamic_classes)} dynamic classes")
            return dynamic_classes
        except Exception as e:
            logger.error(f"Dynamic class creation failed: {e}")
            raise

    def generate_source_code(self, normalized_schema: Dict[str, Any]) -> str:
        """
        Generate static Python source code from normalized schema.

        Args:
            normalized_schema: Normalized schema dictionary

        Returns:
            Generated Python source code as string

        Raises:
            ValueError: If source generation fails
        """
        try:
            source = generate_static_source(normalized_schema)
            classes_count = len(normalized_schema.get("classes", {}))
            logger.debug(f"Generated source code for {classes_count} classes")
            return source
        except Exception as e:
            logger.error(f"Source code generation failed: {e}")
            raise

    def parse_source_code(self, source_code: str, add_metadata_defaults: bool = True) -> Dict[str, Any]:
        """
        Parse Pydantic source code back into schema dictionary.

        Args:
            source_code: Python source code containing Pydantic models
            add_metadata_defaults: Whether to add default metadata fields for compatibility

        Returns:
            Parsed schema dictionary

        Raises:
            ValueError: If parsing fails
            SyntaxError: If source code has syntax errors
        """
        try:
            parsed_schema = parse_pydantic_source(source_code, add_metadata_defaults=add_metadata_defaults)
            classes_count = len(parsed_schema.get("classes", {}))
            logger.debug(f"Parsed source code into {classes_count} classes")
            return parsed_schema
        except Exception as e:
            logger.error(f"Source code parsing failed: {e}")
            raise

    def generate_from_dict(
        self,
        raw_schema: Dict[str, Any],
        create_dynamic: bool = True,
    ) -> CodeGenerationResult:
        """
        Complete code generation workflow from schema dictionary.

        Args:
            raw_schema: Raw schema dictionary
            create_dynamic: Whether to create dynamic classes

        Returns:
            CodeGenerationResult with generated source code as string

        Raises:
            ValueError: If schema processing fails
        """
        # Normalize schema
        normalized_schema, was_wrapped = self.normalize_schema(raw_schema)

        # Validate schema
        validated_schema = self.validate_schema(normalized_schema)

        # Create dynamic classes if requested
        dynamic_classes = None
        if create_dynamic:
            dynamic_classes = self.create_dynamic_classes(validated_schema)

        # Generate source code as string
        generated_source = self.generate_source_code(normalized_schema)

        logger.info(f"Generated source code string with {len(generated_source)} characters")

        # Count generated classes
        classes_count = len(normalized_schema.get("classes", {}))

        return CodeGenerationResult(
            generated_source=generated_source,
            classes_generated=classes_count,
            dynamic_classes=dynamic_classes,
            original_schema=raw_schema,
            normalized_schema=normalized_schema,
            was_wrapped=was_wrapped,
        )

    def parse_from_source(
        self,
        source_code: str,
        normalize: bool = True,
        add_metadata_defaults: bool = True,
    ) -> ParseResult:
        """
        Complete parsing workflow from source code to schema dictionary.

        Args:
            source_code: Python source code containing Pydantic models
            normalize: Whether to normalize the parsed schema
            add_metadata_defaults: Whether to add default metadata fields for compatibility

        Returns:
            ParseResult with parsed schema dictionary

        Raises:
            ValueError: If parsing fails
            SyntaxError: If source code has syntax errors
        """
        # Parse source code with metadata control
        parsed_schema = self.parse_source_code(source_code, add_metadata_defaults=add_metadata_defaults)

        # Normalize if requested
        normalized_schema = None
        was_wrapped = False
        if normalize:
            normalized_schema, was_wrapped = self.normalize_schema(parsed_schema)

        logger.info(f"Parsed source code into schema with {len(parsed_schema.get('classes', {}))} classes")

        return ParseResult(
            parsed_schema=parsed_schema,
            classes_found=len(parsed_schema.get("classes", {})),
            original_source=source_code,
            normalized_schema=normalized_schema,
            was_wrapped=was_wrapped,
        )

    def test_roundtrip(self, schema: Dict[str, Any]) -> bool:
        """
        Test schema -> source -> schema roundtrip conversion.

        Args:
            schema: Original schema dictionary

        Returns:
            True if roundtrip is successful

        Raises:
            ValueError: If roundtrip fails
        """
        try:
            # Generate source from schema (skip validation for schemas from parsed source)
            # Check if this schema lacks metadata (likely from parsed source)
            skip_validation = "executive_summary" not in schema and "classes" in schema

            if skip_validation:
                # For parsed schemas without metadata, generate directly without validation
                normalized_schema, was_wrapped = self.normalize_schema(schema)
                generated_source = self.generate_source_code(normalized_schema)
            else:
                # Normal generation with validation
                result = self.generate_from_dict(schema, create_dynamic=False)
                generated_source = result.generated_source

            # Parse source back to schema (without metadata defaults for clean comparison)
            parse_result = self.parse_from_source(generated_source, normalize=True, add_metadata_defaults=False)
            parsed_schema = parse_result.parsed_schema

            # Ensure both schemas are in the same format for comparison
            # Normalize the original to match the parsed format (with "classes" wrapper)
            original_normalized, _ = self.normalize_schema(schema)

            # Clean both schemas for comparison (removes null values, standardizes format)
            original_clean = self._clean_schema_for_comparison(original_normalized)
            parsed_clean = self._clean_schema_for_comparison(parsed_schema)

            # Compare cleaned schemas
            success = self._deep_compare_schemas(original_clean, parsed_clean)

            if success:
                logger.info("Roundtrip test successful")
            else:
                logger.warning("Roundtrip test failed - schemas differ")
                # For debugging, show the difference
                logger.debug(f"Original clean: {original_clean}")
                logger.debug(f"Parsed clean: {parsed_clean}")

            return success

        except Exception as e:
            logger.error(f"Roundtrip test failed: {e}")
            raise ValueError(f"Roundtrip test failed: {e}") from e

    def _deep_compare_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> bool:
        """Deep compare two cleaned schemas for semantic equivalence."""
        import json

        # Convert both to JSON strings and compare
        # This handles ordering and formatting differences
        try:
            json1 = json.dumps(schema1, sort_keys=True, separators=(",", ":"))
            json2 = json.dumps(schema2, sort_keys=True, separators=(",", ":"))

            if json1 == json2:
                return True
            else:
                logger.debug("Schema JSON representations differ:")
                logger.debug(f"Schema 1: {json1}")
                logger.debug(f"Schema 2: {json2}")
                return False

        except Exception as e:
            logger.error(f"Error comparing schemas: {e}")
            return False

    def _clean_schema_for_comparison(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean schema by removing null values and normalizing structure for comparison."""
        # Ensure we have the classes wrapper
        if "classes" in schema:
            classes = schema["classes"]
        else:
            classes = schema

        cleaned = {"classes": {}}

        # Note: We only include class definitions in the comparison
        # Metadata fields (executive_summary, assumptions) are not preserved in source -> schema conversion
        # So we exclude them from the comparison to allow successful roundtrips

        for class_name, class_info in classes.items():
            cleaned_class = {"fields": {}, "base_classes": class_info.get("base_classes", ["BaseModel"])}

            # Add class description if present and not empty
            if class_info.get("class_description"):
                cleaned_class["class_description"] = class_info["class_description"]

            # Clean fields
            for field_name, field_info in class_info.get("fields", {}).items():
                cleaned_field = {}

                # Always include type and required
                cleaned_field["type"] = field_info.get("type", "Any")
                cleaned_field["required"] = field_info.get("required", True)

                # Only include default if it's meaningful (not null for required fields)
                if "default" in field_info:
                    default_val = field_info["default"]
                    if not (default_val is None and cleaned_field["required"]):
                        cleaned_field["default"] = default_val

                # Include description if present
                if field_info.get("description"):
                    cleaned_field["description"] = field_info["description"]

                # Only include constraints if they exist and are not null/empty
                constraints = field_info.get("constraints")
                if constraints and constraints != {} and constraints is not None:
                    cleaned_field["constraints"] = constraints

                # Include alias if present
                if field_info.get("alias"):
                    cleaned_field["alias"] = field_info["alias"]

                cleaned_class["fields"][field_name] = cleaned_field

            cleaned["classes"][class_name] = cleaned_class

        return cleaned

    def _compare_cleaned_schemas(self, original: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
        """Compare two cleaned schema dictionaries."""
        orig_classes = original["classes"]
        parsed_classes = parsed["classes"]

        if set(orig_classes.keys()) != set(parsed_classes.keys()):
            logger.debug(f"Class names differ: {set(orig_classes.keys())} vs {set(parsed_classes.keys())}")
            return False

        for class_name in orig_classes.keys():
            orig_class = orig_classes[class_name]
            parsed_class = parsed_classes[class_name]

            # Compare base classes
            if orig_class.get("base_classes", []) != parsed_class.get("base_classes", []):
                logger.debug(f"Base classes differ for {class_name}")
                return False

            # Compare class descriptions (optional)
            orig_desc = orig_class.get("class_description", "")
            parsed_desc = parsed_class.get("class_description", "")
            if orig_desc != parsed_desc:
                logger.debug(f"Class descriptions differ for {class_name}")
                # Don't fail on description differences

            # Compare fields
            if not self._compare_cleaned_fields(orig_class["fields"], parsed_class["fields"], class_name):
                return False

        return True

    def _compare_cleaned_fields(
        self, orig_fields: Dict[str, Any], parsed_fields: Dict[str, Any], class_name: str
    ) -> bool:
        """Compare cleaned field definitions."""
        # Create mapping of parsed fields by alias (if present) or name
        parsed_by_alias: Dict[str, Dict[str, Any]] = {}

        for field_name, field_info in parsed_fields.items():
            alias = field_info.get("alias", field_name)
            parsed_by_alias[alias] = field_info

        # Check each original field
        for orig_name, orig_info in orig_fields.items():
            if orig_name not in parsed_by_alias:
                logger.debug(f"Field {orig_name} in class {class_name} not found in parsed schema")
                return False

            parsed_info = parsed_by_alias[orig_name]

            # Compare all field properties
            for key in ["type", "required", "default", "description", "constraints"]:
                orig_val = orig_info.get(key)
                parsed_val = parsed_info.get(key)

                # Handle constraints specially - both empty dicts should be equal
                if key == "constraints":
                    orig_val = orig_val or {}
                    parsed_val = parsed_val or {}

                if orig_val != parsed_val:
                    logger.debug(f"Field {orig_name} in {class_name}: {key} differs - {orig_val} vs {parsed_val}")
                    return False

        return True

    def _compare_schemas(self, original: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
        """Compare two schema dictionaries for equivalence."""
        # Normalize both schemas to have "classes" wrapper
        orig_normalized, _ = self.normalize_schema(original)
        parsed_normalized, _ = self.normalize_schema(parsed)

        orig_classes = orig_normalized["classes"]
        parsed_classes = parsed_normalized["classes"]

        if set(orig_classes.keys()) != set(parsed_classes.keys()):
            logger.debug(f"Class names differ: {set(orig_classes.keys())} vs {set(parsed_classes.keys())}")
            return False

        for class_name in orig_classes.keys():
            if not self._compare_class_definitions(orig_classes[class_name], parsed_classes[class_name], class_name):
                return False

        return True

    def _compare_class_definitions(self, original: Dict[str, Any], parsed: Dict[str, Any], class_name: str) -> bool:
        """Compare two class definitions for equivalence."""
        orig_fields = original.get("fields", {})
        parsed_fields = parsed.get("fields", {})

        # Handle field name mapping for sanitized identifiers
        if not self._compare_fields(orig_fields, parsed_fields, class_name):
            return False

        # Compare descriptions (optional)
        orig_desc = original.get("class_description", "").strip()
        parsed_desc = parsed.get("class_description", "").strip()
        if orig_desc != parsed_desc:
            logger.debug(f"Class {class_name} descriptions differ")
            # Don't fail on description differences - they're not critical

        return True

    def _compare_fields(self, orig_fields: Dict[str, Any], parsed_fields: Dict[str, Any], class_name: str) -> bool:
        """Compare field definitions, handling aliases and sanitization."""
        # Create mapping of parsed fields by alias (if present) or name
        parsed_by_alias: Dict[str, Dict[str, Any]] = {}

        for field_name, field_info in parsed_fields.items():
            alias = field_info.get("alias", field_name)
            parsed_by_alias[alias] = field_info

        # Check each original field
        for orig_name, orig_info in orig_fields.items():
            if orig_name not in parsed_by_alias:
                logger.debug(f"Field {orig_name} in class {class_name} not found in parsed schema")
                return False

            parsed_info = parsed_by_alias[orig_name]
            if not self._compare_field_info(orig_info, parsed_info, orig_name, class_name):
                return False

        return True

    def _compare_field_info(
        self, original: Dict[str, Any], parsed: Dict[str, Any], field_name: str, class_name: str
    ) -> bool:
        """Compare individual field information."""
        # Normalize both field infos for comparison
        orig_normalized = self._normalize_field_for_comparison(original)
        parsed_normalized = self._normalize_field_for_comparison(parsed)

        # Check required fields
        for key in ["type", "required"]:
            if orig_normalized.get(key) != parsed_normalized.get(key):
                logger.debug(
                    f"Field {field_name} in {class_name}: {key} differs - {orig_normalized.get(key)} vs {parsed_normalized.get(key)}"
                )
                return False

        # Check optional fields with normalization
        for key in ["default", "description"]:
            orig_val = orig_normalized.get(key)
            parsed_val = parsed_normalized.get(key)
            if orig_val != parsed_val:
                logger.debug(f"Field {field_name} in {class_name}: {key} differs - {orig_val} vs {parsed_val}")
                return False

        # Special handling for constraints
        orig_constraints = orig_normalized.get("constraints", {})
        parsed_constraints = parsed_normalized.get("constraints", {})
        if orig_constraints != parsed_constraints:
            logger.debug(
                f"Field {field_name} in {class_name}: constraints differ - {orig_constraints} vs {parsed_constraints}"
            )
            # Be more lenient with constraints as formatting may differ
            if not (orig_constraints == {} and parsed_constraints == {}):
                return False

        return True

    def _normalize_field_for_comparison(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field info for comparison by removing null values and standardizing format."""
        normalized = {}

        # Copy non-null values, handling special cases
        for key, value in field_info.items():
            if key == "default" and value is None and not field_info.get("required", True):
                # Preserve explicit None defaults for optional fields
                normalized[key] = None
            elif key == "constraints" and (value is None or value == {}):
                # Skip null or empty constraints
                continue
            elif value is not None:
                normalized[key] = value

        # Ensure required field has default value if not specified
        if "required" not in normalized:
            normalized["required"] = True

        return normalized

    def set_schema_class(self, schema_class: Type[BaseModel]) -> None:
        """
        Change the schema validation class.

        Args:
            schema_class: New schema validation class
        """
        self.schema_class = schema_class
        logger.info(f"Schema class changed to: {schema_class.__name__}")

    def validate_schema_structure(self, schema: Dict[str, Any]) -> bool:
        """
        Validate basic schema structure without full validation.

        Args:
            schema: Schema dictionary to validate

        Returns:
            True if structure is valid

        Raises:
            ValueError: If structure is invalid
        """
        try:
            normalized, _ = self.normalize_schema(schema)
            # Basic structure check
            if "classes" not in normalized:
                raise ValueError("Normalized schema missing 'classes' key")

            classes = normalized["classes"]
            if not isinstance(classes, dict):
                raise ValueError("'classes' must be a dictionary")

            for class_name, class_info in classes.items():
                if not isinstance(class_info, dict):
                    raise ValueError(f"Class '{class_name}' definition must be a dictionary")
                if "fields" not in class_info:
                    raise ValueError(f"Class '{class_name}' missing 'fields' key")
                if not isinstance(class_info["fields"], dict):
                    raise ValueError(f"Class '{class_name}' fields must be a dictionary")

            logger.debug("Schema structure validation passed")
            return True

        except Exception as e:
            logger.error(f"Schema structure validation failed: {e}")
            raise
