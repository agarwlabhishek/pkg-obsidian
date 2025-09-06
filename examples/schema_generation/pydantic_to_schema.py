#!/usr/bin/env python3
"""
Pydantic to Schema CLI - generates JSON schemas from Pydantic model files.
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


def format_json_compact(obj, indent=2):
    """Format JSON with compact single-line formatting for simple objects/arrays."""
    import json

    def should_be_compact(obj):
        """Determine if an object should use compact single-line formatting."""
        if isinstance(obj, dict):
            if not obj or len(obj) > 3:
                return False
            # Check if all values are simple AND short
            for v in obj.values():
                if isinstance(v, str) and len(v) > 50:  # Long strings stay multi-line
                    return False
                if not isinstance(v, (str, int, float, bool, type(None))):
                    return False
            return True
        elif isinstance(obj, list):
            if not obj or len(obj) > 3:
                return False
            # All items must be simple primitives
            return all(isinstance(item, (str, int, float, bool, type(None))) for item in obj)
        return False

    def compact_format(obj, current_indent=0, is_root=False):
        if isinstance(obj, dict):
            if not obj:
                return "{}"

            if should_be_compact(obj):
                items = [f'"{k}": {json.dumps(v)}' for k, v in obj.items()]
                return "{ " + ", ".join(items) + " }"
            else:
                indent_str = " " * current_indent
                next_indent_str = " " * (current_indent + indent)
                items = []

                for k, v in obj.items():
                    formatted_v = compact_format(v, current_indent + indent)
                    items.append(f'{next_indent_str}"{k}": {formatted_v}')

                if is_root:
                    # Add empty lines between root-level classes
                    return "{\n" + ",\n\n".join(items) + "\n" + indent_str + "}"
                else:
                    return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            if should_be_compact(obj):
                items = [json.dumps(item) for item in obj]
                return "[" + ", ".join(items) + "]"
            else:
                indent_str = " " * current_indent
                next_indent_str = " " * (current_indent + indent)
                items = []
                for item in obj:
                    formatted_item = compact_format(item, current_indent + indent)
                    items.append(f"{next_indent_str}{formatted_item}")
                return "[\n" + ",\n".join(items) + "\n" + indent_str + "]"
        else:
            return json.dumps(obj)

    # Check if this is a root object with class definitions
    is_root_classes = isinstance(obj, dict) and any(isinstance(v, dict) and "fields" in v for v in obj.values())

    result = compact_format(obj, 0, is_root_classes)

    # Ensure the result ends with a newline
    if not result.endswith("\n"):
        result += "\n"

    return result


def load_python_file(file_path: Path) -> str:
    """Load Python source from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Python file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to read Python file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON schemas from Pydantic model files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s models.py                    # Parse and print to stdout
  %(prog)s models.py -o results.json    # Save to results.json
  %(prog)s models.py --normalize        # Normalize parsed schema
  %(prog)s models.py --log-level DEBUG  # Verbose logging
  %(prog)s models.py --no-metadata      # Don't add metadata defaults
        """,
    )

    parser.add_argument("python_file", type=Path, help="Input Python file with Pydantic models")
    parser.add_argument("-o", "--output", type=Path, help="Output file for generated schema (default: print to stdout)")
    parser.add_argument("--normalize", action="store_true", help="Normalize parsed schema")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't add default metadata fields (executive_summary, assumptions) to parsed schema",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    parser.add_argument("--test-roundtrip", action="store_true", help="Test roundtrip conversion after parsing")

    args = parser.parse_args()

    # Configure logging
    logger = build_logger(args.log_level)

    # Validate input file
    if not args.python_file.exists():
        print(f"Error: Python file not found: {args.python_file}", file=sys.stderr)
        return 1

    try:
        # Load Python source from file
        source_code = load_python_file(args.python_file)

        # Initialize generator
        generator = SchemaCodeGenerator()

        # Parse source code
        # Add metadata defaults unless --no-metadata is specified
        add_metadata = not args.no_metadata
        result = generator.parse_from_source(source_code, normalize=args.normalize, add_metadata_defaults=add_metadata)

        # Output results - use same format as input when possible
        output_schema = result.parsed_schema
        if not args.normalize and "classes" in output_schema:
            # If not normalizing, output in "bare" format (classes at root level)
            # to match common input format - but only if no metadata is present
            if not add_metadata or (
                not output_schema.get("executive_summary") and not output_schema.get("assumptions")
            ):
                output_schema = output_schema["classes"]

        # Use compact JSON formatting to match typical input format
        schema_json = format_json_compact(output_schema, indent=2)

        if args.output:
            # Write to file
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(schema_json, encoding="utf-8")
            print(f"Parsed {result.classes_found} classes -> {args.output}")
        else:
            # Print to stdout
            print(schema_json)

        # Test roundtrip conversion if requested
        if args.test_roundtrip:
            print(f"\nTesting roundtrip conversion:", file=sys.stderr)
            try:
                success = generator.test_roundtrip(result.parsed_schema)
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
