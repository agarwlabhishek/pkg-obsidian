#!/usr/bin/env python3
"""
Schema to Pydantic CLI - generates Pydantic models from JSON schema files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from da3_obsidian.pydantic_schema_codegen import SchemaCodeGenerator, build_logger
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def load_schema_file(file_path: Path) -> Dict[str, Any]:
    """Load schema from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read schema file {file_path}: {e}")


def ensure_schema_metadata(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure schema has required metadata fields for validation.

    Args:
        schema: Raw schema dictionary

    Returns:
        Schema with metadata fields (added if missing)
    """
    # Check if this is a bare schema (classes at root) or wrapped schema
    if "classes" not in schema:
        # Assume it's a bare schema - wrap it
        schema = {"classes": schema}

    # Add default metadata if missing
    if "executive_summary" not in schema:
        schema["executive_summary"] = "Auto-generated Pydantic models from JSON schema"

    if "assumptions" not in schema:
        schema["assumptions"] = []

    return schema


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pydantic models from JSON schema files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s schema.json                    # Generate and print to stdout
  %(prog)s schema.json -o results.py      # Save to results.py
  %(prog)s schema.json --no-dynamic       # Skip dynamic class creation
  %(prog)s schema.json --test-dynamic     # Test dynamic classes
  %(prog)s schema.json --log-level DEBUG  # Verbose logging
        """,
    )

    parser.add_argument("schema_file", type=Path, help="Input JSON schema file")
    parser.add_argument("-o", "--output", type=Path, help="Output file for generated code (default: print to stdout)")
    parser.add_argument("--no-dynamic", action="store_true", help="Skip dynamic class creation (faster)")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    parser.add_argument("--test-dynamic", action="store_true", help="Test dynamic classes after generation")
    parser.add_argument("--test-roundtrip", action="store_true", help="Test roundtrip conversion after generation")

    args = parser.parse_args()

    # Configure logging
    logger = build_logger(args.log_level)

    # Validate input file
    if not args.schema_file.exists():
        print(f"Error: Schema file not found: {args.schema_file}", file=sys.stderr)
        return 1

    try:
        # Load schema from file
        raw_schema = load_schema_file(args.schema_file)

        # Ensure schema has metadata for validation
        schema = ensure_schema_metadata(raw_schema)

        # Initialize generator
        generator = SchemaCodeGenerator()

        # Generate code
        result = generator.generate_from_dict(schema, create_dynamic=not args.no_dynamic)

        # Output results
        if args.output:
            # Write to file
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result.generated_source, encoding="utf-8")
            print(f"Generated {result.classes_generated} classes -> {args.output}")
        else:
            # Print to stdout
            print(result.generated_source)

        # Test dynamic classes if requested
        if args.test_dynamic and result.dynamic_classes:
            print(f"\nTesting {len(result.dynamic_classes)} dynamic classes:", file=sys.stderr)
            for class_name, cls in result.dynamic_classes.items():
                try:
                    # Get field information to create valid test data
                    fields = schema["classes"].get(class_name, {}).get("fields", {})

                    # Build minimal valid data for required fields
                    test_data = {}
                    for field_name, field_info in fields.items():
                        if field_info.get("required", True):
                            # Provide minimal valid values based on type
                            field_type = field_info.get("type", "str")
                            if "str" in field_type:
                                test_data[field_name] = "test"
                            elif "int" in field_type:
                                test_data[field_name] = 0
                            elif "float" in field_type:
                                test_data[field_name] = 0.0
                            elif "bool" in field_type:
                                test_data[field_name] = False
                            elif "datetime" in field_type:
                                from datetime import datetime

                                test_data[field_name] = datetime.now()
                            elif "Dict" in field_type:
                                test_data[field_name] = {}
                            elif "List" in field_type:
                                test_data[field_name] = []
                            else:
                                # For complex types, try to instantiate nested class
                                if field_type in result.dynamic_classes:
                                    # Recursively create nested instance
                                    test_data[field_name] = result.dynamic_classes[field_type]()

                    # Try to instantiate with test data
                    instance = cls(**test_data)
                    print(f"  ✓ {class_name}: instantiated successfully", file=sys.stderr)
                except Exception as e:
                    print(f"  ✗ {class_name}: {e}", file=sys.stderr)

        # Test roundtrip if requested
        if args.test_roundtrip:
            print(f"\nTesting roundtrip conversion:", file=sys.stderr)
            try:
                # Use the original schema (without added metadata) for roundtrip
                success = generator.test_roundtrip(raw_schema)
                if success:
                    print(f"  ✓ Roundtrip test passed", file=sys.stderr)
                else:
                    print(f"  ✗ Roundtrip test failed", file=sys.stderr)
            except Exception as e:
                print(f"  ✗ Roundtrip test error: {e}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
