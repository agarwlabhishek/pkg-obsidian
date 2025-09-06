"""
Static Python source code generation from schema definitions.
"""

from __future__ import annotations
import keyword
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Conservative, always-on imports in the emitted file
_ALWAYS_TYPING = "from typing import Any, Dict, Optional, List, Union, Literal"
_ALWAYS_PYDANTIC = "from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict"


@dataclass(frozen=True)
class FieldRender:
    """Represents a rendered field definition."""

    name: str
    annotation: str
    field_args: str  # inside Field(...)


def _repr_arg(value: Any) -> str:
    """Convert value to its Python representation string."""
    return repr(value)


def sanitize_identifier(name: str) -> Tuple[str, str | None]:
    """
    Sanitize a string to be a valid Python identifier.

    Args:
        name: Original name to sanitize

    Returns:
        Tuple of (safe_name, alias) where alias is the original name if different
    """
    # Replace non-alphanumeric characters with underscores
    safe = "".join(ch if (ch == "_" or ch.isalnum()) else "_" for ch in name)

    # Ensure doesn't start with digit
    if safe and safe[0].isdigit():
        safe = f"_{safe}"

    # Ensure not empty
    if not safe:
        safe = "field_"

    # Handle reserved keywords
    if keyword.iskeyword(safe):
        safe = f"{safe}_"

    # Return alias if name was changed
    if not safe.isidentifier() or safe != name:
        return safe, name

    return safe, None


def render_field(name: str, spec: Dict[str, Any], used_names: set[str]) -> FieldRender:
    """
    Render a field definition from specification.

    Args:
        name: Field name
        spec: Field specification dictionary
        used_names: Set of already used field names (updated in place)

    Returns:
        FieldRender with safe name, annotation, and field arguments
    """
    annotation = spec.get("type") or "Any"
    safe_name, alias = sanitize_identifier(name)

    # Ensure unique name
    base_safe = safe_name
    counter = 1
    while safe_name in used_names:
        counter += 1
        safe_name = f"{base_safe}{counter}"
    used_names.add(safe_name)

    # Determine default value
    if spec.get("default") is not None:
        default_arg = _repr_arg(spec["default"])
    elif not spec.get("required", True):
        default_arg = "None"
    else:
        default_arg = "..."

    # Build Field() arguments
    kwargs: List[str] = [default_arg]

    # Add description if present
    if desc := spec.get("description"):
        kwargs.append(f"description={_repr_arg(desc)}")

    # Add alias if name was changed
    if alias is not None:
        kwargs.append(f"alias={_repr_arg(alias)}")

    # Add constraint parameters
    for key, val in (spec.get("constraints") or {}).items():
        kwargs.append(f"{key}={_repr_arg(val)}")

    return FieldRender(name=safe_name, annotation=annotation, field_args=", ".join(kwargs))


def generate_static_source(schema_json: Dict[str, Any]) -> str:
    """
    Generate static Python source code from schema JSON.

    Args:
        schema_json: Schema dictionary with 'classes' key

    Returns:
        Generated Python source code as string

    Raises:
        ValueError: If schema format is invalid
    """
    classes = schema_json.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("schema_json['classes'] must be a non-empty dict")

    # Generate timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    # Start building source lines
    lines: List[str] = [
        "# Auto-generated using DA3 Obsidian — DO NOT EDIT",
        f"# Generated at: {timestamp}",
    ]

    # Add executive summary if present
    if executive_summary := schema_json.get("executive_summary"):
        lines.append("#")
        lines.append("# Executive Summary:")
        # Wrap executive summary text
        for line in executive_summary.split("\n"):
            if line.strip():
                lines.append(f"# {line}")

    # Add assumptions if present
    if assumptions := schema_json.get("assumptions"):
        lines.append("#")
        lines.append("# Assumptions:")
        for assumption in assumptions:
            lines.append(f"# - {assumption}")

    lines.extend(
        [
            "",
            "from __future__ import annotations",
            "",
            _ALWAYS_TYPING,
            _ALWAYS_PYDANTIC,
            "",
        ]
    )

    logger.debug(f"Generating source for {len(classes)} classes")

    # Generate each class
    for cls_name, class_info in classes.items():
        logger.debug(f"Generating class: {cls_name}")

        # Get base classes
        base_list = class_info.get("base_classes") or ["BaseModel"]
        bases = ", ".join(base_list)

        # Get class description
        class_desc = class_info.get("class_description") or ""

        # Class definition line
        lines.append(f"class {cls_name}({bases}):")

        # Class docstring if available
        if class_desc:
            lines.append(f'    """{class_desc}"""')

        # Model configuration
        lines.append("    # Model configuration: ignore unknown fields; allow population by field name")
        lines.append("    model_config = ConfigDict(extra='ignore', populate_by_name=True)")

        # Process fields
        fields = class_info.get("fields") or {}
        if not fields:
            lines.append("    pass")
            lines.append("")
            continue

        used_names: set[str] = set()

        for field_name, field_spec in fields.items():
            field_render = render_field(field_name, field_spec, used_names)
            field_line = f"    {field_render.name}: {field_render.annotation} = Field({field_render.field_args})"
            lines.append(field_line)

        lines.append("")

    source = "\n".join(lines)
    logger.info(f"Generated {len(lines)} lines of source code")

    return source
