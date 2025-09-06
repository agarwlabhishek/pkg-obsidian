"""
Parse Pydantic source code back into schema dictionary format.
"""

from __future__ import annotations
import ast
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class PydanticSourceParser:
    """Parser for extracting schema information from Pydantic source code."""

    def __init__(self):
        self.parsed_classes: Dict[str, Dict[str, Any]] = {}

    def parse_source(self, source_code: str, add_metadata_defaults: bool = True) -> Dict[str, Any]:
        """
        Parse Pydantic source code and extract schema dictionary.

        Args:
            source_code: Python source code containing Pydantic models
            add_metadata_defaults: Whether to add default metadata fields for compatibility

        Returns:
            Schema dictionary in the format expected by SchemaCodeGenerator

        Raises:
            ValueError: If source code cannot be parsed
            SyntaxError: If source code has syntax errors
        """
        try:
            tree = ast.parse(source_code)
            self.parsed_classes = {}

            # Find all class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if self._is_pydantic_model(node):
                        class_info = self._extract_class_info(node)
                        self.parsed_classes[node.name] = class_info

            if not self.parsed_classes:
                raise ValueError("No Pydantic model classes found in source code")

            logger.info(f"Parsed {len(self.parsed_classes)} Pydantic classes")

            # Build the schema
            result = {"classes": self.parsed_classes}

            # Add default metadata if requested (for compatibility with StructuredSchemaResponse)
            if add_metadata_defaults:
                result["executive_summary"] = "Auto-generated schema from parsed Pydantic source code"
                result["assumptions"] = []

            return result

        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse source code: {e}")
            raise ValueError(f"Source code parsing failed: {e}") from e

    def _is_pydantic_model(self, class_node: ast.ClassDef) -> bool:
        """Check if a class definition is a Pydantic model."""
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                return True
            elif isinstance(base, ast.Attribute):
                # Handle cases like pydantic.BaseModel
                if base.attr == "BaseModel":
                    return True
        return False

    def _extract_class_info(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information from AST node."""
        class_info: Dict[str, Any] = {
            "fields": {},
        }

        # Extract field definitions first
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name, field_info = self._extract_field_info(node)
                if field_info:  # Only add if we successfully parsed the field
                    class_info["fields"][field_name] = field_info

        # Add class description next (to match original ordering)
        docstring = ast.get_docstring(class_node)
        if docstring:
            class_info["class_description"] = docstring

        # Add base classes last
        class_info["base_classes"] = self._extract_base_classes(class_node)

        logger.debug(f"Extracted class {class_node.name} with {len(class_info['fields'])} fields")
        return class_info

    def _extract_base_classes(self, class_node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        bases = []
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle dotted names like pydantic.BaseModel
                bases.append(self._get_attribute_name(base))
        return bases if bases else ["BaseModel"]

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name from AST node."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return node.attr

    def _extract_field_info(self, ann_assign: ast.AnnAssign) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Extract field information from annotated assignment."""
        field_name = ann_assign.target.id
        field_info: Dict[str, Any] = {}

        # Extract type annotation
        type_annotation = self._extract_type_annotation(ann_assign.annotation)
        field_info["type"] = type_annotation

        # Extract Field() call information if present
        if ann_assign.value:
            field_params = self._extract_field_call(ann_assign.value)
            if field_params is not None:
                field_info.update(field_params)
            else:
                # Direct assignment (not Field() call)
                field_info["default"] = self._extract_literal_value(ann_assign.value)
                field_info["required"] = False
        else:
            # No default value, field is required
            field_info["required"] = True

        # Set required to True if not explicitly set and no default
        if "required" not in field_info and "default" not in field_info:
            field_info["required"] = True

        return field_name, field_info

    def _extract_type_annotation(self, annotation: ast.AST) -> str:
        """Extract type annotation as string."""
        try:
            return ast.get_source_segment("", annotation) or self._ast_to_string(annotation)
        except:
            return self._ast_to_string(annotation)

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._ast_to_string(node.value)}[{self._ast_to_string(node.slice)}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._ast_to_string(elt) for elt in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.List):
            elements = [self._ast_to_string(elt) for elt in node.elts]
            return f"[{', '.join(elements)}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Handle Union types written as X | Y
            return f"{self._ast_to_string(node.left)} | {self._ast_to_string(node.right)}"
        else:
            # Fallback for complex expressions
            try:
                return ast.unparse(node)
            except:
                return "Any"

    def _extract_field_call(self, value_node: ast.AST) -> Optional[Dict[str, Any]]:
        """Extract parameters from Field() call."""
        if not isinstance(value_node, ast.Call):
            return None

        # Check if it's a Field() call
        if isinstance(value_node.func, ast.Name) and value_node.func.id == "Field":
            return self._parse_field_arguments(value_node)

        return None

    def _parse_field_arguments(self, call_node: ast.Call) -> Dict[str, Any]:
        """Parse arguments from Field() call."""
        field_info: Dict[str, Any] = {}
        constraints: Dict[str, Any] = {}

        # Process positional arguments
        if call_node.args:
            # First positional argument is default value
            default_value = self._extract_literal_value(call_node.args[0])
            if default_value == "...":
                field_info["required"] = True
            else:
                field_info["required"] = False
                field_info["default"] = default_value

        # Process keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg is None:
                continue

            value = self._extract_literal_value(keyword.value)

            if keyword.arg == "description":
                field_info["description"] = value
            elif keyword.arg == "alias":
                # Store original field name as alias if different
                field_info["alias"] = value
            elif keyword.arg in ["default", "default_factory"]:
                field_info["default"] = value
                field_info["required"] = False
            elif keyword.arg in [
                "min_length",
                "max_length",
                "pattern",
                "gt",
                "ge",
                "lt",
                "le",
                "multiple_of",
                "min_items",
                "max_items",
            ]:
                constraints[keyword.arg] = value

        # Add constraints if any (don't add empty constraints)
        if constraints:
            field_info["constraints"] = constraints

        return field_info

    def _extract_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Ellipsis):
            return "..."
        elif isinstance(node, ast.Constant):
            # Handle ellipsis constant in Python 3.8+
            if node.value is ...:
                return "..."
            return node.value
        elif isinstance(node, ast.Name):
            if node.id == "None":
                return None
            elif node.id == "True":
                return True
            elif node.id == "False":
                return False
            else:
                return node.id  # Return as string for non-literal names
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.List):
            return [self._extract_literal_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is not None:
                    key = self._extract_literal_value(key_node)
                    value = self._extract_literal_value(value_node)
                    result[key] = value
            return result
        else:
            # For complex expressions, return string representation
            try:
                return ast.unparse(node)
            except:
                return str(node)


def parse_pydantic_source(source_code: str, add_metadata_defaults: bool = True) -> Dict[str, Any]:
    """
    Parse Pydantic source code and return schema dictionary.

    Args:
        source_code: Python source code containing Pydantic models
        add_metadata_defaults: Whether to add default metadata fields for compatibility

    Returns:
        Schema dictionary compatible with SchemaCodeGenerator

    Raises:
        ValueError: If parsing fails
        SyntaxError: If source code has syntax errors
    """
    parser = PydanticSourceParser()
    return parser.parse_source(source_code, add_metadata_defaults=add_metadata_defaults)
