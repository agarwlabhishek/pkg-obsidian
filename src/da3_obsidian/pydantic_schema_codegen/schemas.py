"""
Response schemas for Pydantic code generation.

This module defines the response schemas used by the schema generation workflow,
including structured schema definitions and Pydantic field specifications.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class PydanticFieldDefinition(BaseModel):
    """Structured definition for a Pydantic field."""

    type: str = Field(
        ...,
        description="Python type for the field - supports basic types (str, int, float, bool, list, dict) and complex types (Optional[str], List[int], Union[str, int], etc.)",
    )
    required: bool = Field(default=True, description="Whether the field is required")
    default: Optional[Any] = Field(default=None, description="Default value for the field")
    description: Optional[str] = Field(default=None, description="Field description")
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional field constraints (min_length, max_length, minimum, maximum, pattern, ge, le, gt, lt, etc.)",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the type string is reasonable."""
        if not v.strip():
            raise ValueError("Type cannot be empty")
        return v.strip()


class ClassDefinition(BaseModel):
    """Definition for a single Pydantic class."""

    fields: Dict[str, PydanticFieldDefinition] = Field(
        ..., description="Dictionary of field names to their Pydantic field definitions", min_length=1
    )
    class_description: Optional[str] = Field(default=None, description="Description for the Pydantic class")
    base_classes: Optional[List[str]] = Field(
        default=None,
        description="List of base class names this class inherits from (e.g., ['BaseUser', 'TimestampMixin'])",
    )


class StructuredSchemaResponse(BaseModel):
    """Structured response schema that defines one or more Pydantic-compatible classes."""

    classes: Dict[str, ClassDefinition] = Field(
        ..., description="Dictionary of class names to their class definitions", min_length=1
    )

    @field_validator("classes")
    @classmethod
    def validate_class_names(cls, v: Dict[str, ClassDefinition]) -> Dict[str, ClassDefinition]:
        """Validate that all class names are valid Python class identifiers."""
        for class_name in v.keys():
            if not class_name.isidentifier():
                raise ValueError(f"class_name '{class_name}' must be a valid Python identifier")
            if not class_name[0].isupper():
                raise ValueError(f"class_name '{class_name}' should start with an uppercase letter")
        return v

    def to_pydantic_classes(self) -> Dict[str, type[BaseModel]]:
        """Convert this schema to actual Pydantic classes."""
        import re
        from typing import Optional, Union, List as TypingList, Dict as TypingDict

        # Basic type mapping
        basic_type_mapping = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}

        def parse_type_string(type_str: str) -> type:
            """Parse a type string into a Python type."""
            type_str = type_str.strip()

            # Handle Optional[Type]
            optional_match = re.match(r"Optional\[(.+)\]", type_str)
            if optional_match:
                inner_type = parse_type_string(optional_match.group(1))
                return Optional[inner_type]

            # Handle List[Type]
            list_match = re.match(r"List\[(.+)\]", type_str)
            if list_match:
                inner_type = parse_type_string(list_match.group(1))
                return TypingList[inner_type]

            # Handle Dict[KeyType, ValueType]
            dict_match = re.match(r"Dict\[(.+),\s*(.+)\]", type_str)
            if dict_match:
                key_type = parse_type_string(dict_match.group(1))
                value_type = parse_type_string(dict_match.group(2))
                return TypingDict[key_type, value_type]

            # Handle Union[Type1, Type2, ...]
            union_match = re.match(r"Union\[(.+)\]", type_str)
            if union_match:
                # Split by comma but be careful with nested types
                parts = []
                depth = 0
                current = ""
                for char in union_match.group(1):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                    elif char == "," and depth == 0:
                        parts.append(current.strip())
                        current = ""
                        continue
                    current += char
                if current:
                    parts.append(current.strip())

                union_types = [parse_type_string(t) for t in parts]
                return Union[tuple(union_types)]

            # Handle basic types
            if type_str in basic_type_mapping:
                return basic_type_mapping[type_str]

            # Handle references to other classes in this schema
            if type_str in self.classes:
                # This will be resolved after all classes are created
                return type_str

            # Default to str for unknown types
            return str

        result_classes = {}

        # First pass: create all classes without inheritance
        for class_name, class_def in self.classes.items():
            annotations = {}
            class_attrs = {}

            for field_name, field_def in class_def.fields.items():
                field_type = parse_type_string(field_def.type)

                # Handle optional fields
                if not field_def.required and not field_def.type.startswith("Optional["):
                    field_type = Optional[field_type]

                annotations[field_name] = field_type

                # Create Field with constraints if provided
                field_kwargs = {}
                if field_def.default is not None:
                    field_kwargs["default"] = field_def.default
                elif not field_def.required:
                    field_kwargs["default"] = None
                else:
                    field_kwargs["default"] = ...

                if field_def.description:
                    field_kwargs["description"] = field_def.description

                if field_def.constraints:
                    field_kwargs.update(field_def.constraints)

                class_attrs[field_name] = Field(**field_kwargs)

            # Add annotations and docstring
            class_attrs["__annotations__"] = annotations
            if class_def.class_description:
                class_attrs["__doc__"] = class_def.class_description

            # Determine base classes
            base_classes = [BaseModel]
            if class_def.base_classes:
                # For now, we'll just use BaseModel as the base
                # In a real implementation, you'd resolve these references
                pass

            result_classes[class_name] = type(class_name, tuple(base_classes), class_attrs)

        # Second pass: resolve any class references in field types
        for class_name, pydantic_class in result_classes.items():
            annotations = getattr(pydantic_class, "__annotations__", {})
            updated_annotations = {}

            for field_name, field_type in annotations.items():
                if isinstance(field_type, str) and field_type in result_classes:
                    # Replace string reference with actual class
                    updated_annotations[field_name] = result_classes[field_type]
                else:
                    updated_annotations[field_name] = field_type

            if updated_annotations != annotations:
                pydantic_class.__annotations__ = updated_annotations

        return result_classes
