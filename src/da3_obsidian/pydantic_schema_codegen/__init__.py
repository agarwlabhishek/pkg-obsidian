"""
Pydantic schema code generation module for DA3 Obsidian.

This module provides schema-to-Pydantic model generation capabilities
with dynamic class creation, static source code generation, and
source code parsing (inverse operation).
"""

from .generator import CodeGenerationResult, ParseResult, SchemaCodeGenerator
from .core import (
    get_schema_class,
    normalize_schema_payload,
    validate_schema,
    build_dynamic_classes,
)
from .io_utils import build_logger
from .source_generator import generate_static_source
from .source_parser import parse_pydantic_source, PydanticSourceParser
from .schemas import (
    PydanticFieldDefinition,
    ClassDefinition,
    StructuredSchemaResponse,
)

__all__ = [
    "SchemaCodeGenerator",
    "CodeGenerationResult",
    "ParseResult",
    "get_schema_class",
    "normalize_schema_payload",
    "validate_schema",
    "build_dynamic_classes",
    "build_logger",
    "generate_static_source",
    "parse_pydantic_source",
    "PydanticSourceParser",
    "PydanticFieldDefinition",
    "ClassDefinition",
    "StructuredSchemaResponse",
]
