#!/usr/bin/env python3
"""
Advanced forensic analysis script with custom configuration.

Usage:
    python forensics_analyzer.py "sample_documents/sample.pdf" --config custom_config.json --output "results/forensic_analzyer_result.json"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from da3_obsidian.forensic_analysis import ForensicAnalyzer, ForensicConfig


def load_config(config_file: str) -> ForensicConfig:
    """Load configuration from JSON file."""
    with open(config_file, "r") as f:
        config_data = json.load(f)
    return ForensicConfig.from_dict(config_data)


def main():
    """Main function to analyze document with advanced options."""

    parser = argparse.ArgumentParser(description="Advanced forensic analysis with custom configuration")
    parser.add_argument("file_path", help="Path to document or image file")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--hash", help="Trusted SHA256 hash for verification")
    parser.add_argument("--output", help="Output JSON file name")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        print(f"Using custom configuration: {args.config}")
    else:
        config = ForensicConfig()
        print("Using default configuration")

    # Set output file name
    if args.output:
        output_file = args.output
    else:
        file_name = Path(args.file_path).stem
        output_file = f"{file_name}_forensic_analysis.json"

    print(f"Analyzing file: {args.file_path}")
    print(f"Output will be saved to: {output_file}")

    try:
        # Initialize analyzer with configuration
        analyzer = ForensicAnalyzer(config)

        # Perform forensic analysis
        result = analyzer.analyze_document(args.file_path, args.hash)

        # Display detailed results
        print("\n📄 ANALYSIS RESULTS:")
        print(f"File: {os.path.basename(args.file_path)}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Risk Score: {result.risk_score}/100")
        print(f"Total Issues: {len(result.fraud_indicators)}")

        # Show hash verification if provided
        if args.hash:
            hash_info = result.raw_results.get("integrity_analysis", {}).get("file_hashes", {})
            matches = hash_info.get("matches_trusted_hash", False)
            print(f"Hash Verification: {'✅ PASSED' if matches else '❌ FAILED'}")

        # Show configuration used
        print("\n⚙️  CONFIGURATION:")
        config_dict = config.to_dict()
        print(f"  Max PDF Pages: {config_dict['Analysis']['max_pdf_pages']}")
        print(f"  Max Images: {config_dict['Analysis']['max_images']}")
        print(f"  ELA Threshold: {config_dict['Thresholds']['ela_mean_threshold']}")
        print(f"  Copy-Move Threshold: {config_dict['Thresholds']['copy_move_threshold']}")

        # Show categories with details
        issues_by_category = result.get_issues_by_category()
        print("\n📊 DETAILED ANALYSIS:")

        categories = {
            "metadata": "Metadata Issues",
            "image_manipulation": "Image Manipulation",
            "structure_anomalies": "Structure Anomalies",
            "text_inconsistencies": "Text Inconsistencies",
        }

        for category, display_name in categories.items():
            indicators = issues_by_category.get(category, [])
            count = len(indicators)
            status = "🔴" if count > 0 else "✅"
            print(f"  {status} {display_name}: {count}")

            # Show first 2 issues for each category
            if indicators:
                for indicator in indicators[:2]:
                    print(f"    - {indicator}")
                if len(indicators) > 2:
                    print(f"    ... and {len(indicators) - 2} more")

        # Show analysis summary
        summary = analyzer.get_analysis_summary(result)
        print("\n📈 ANALYSIS SUMMARY:")
        print(f"  File Size: {summary['file_size']}")
        print(f"  Risk Assessment: {summary['risk_assessment']}")
        print(f"  Has Issues: {summary['has_issues']}")

        # Save results as JSON
        with open(output_file, "w") as f:
            json.dump(result.raw_results, f, indent=2, default=str)

        print(f"\n💾 Complete results saved to: {output_file}")

        # Show file info
        file_info = result.raw_results.get("file_info", {})
        print("\n📋 FILE INFORMATION:")
        print(f"  Path: {file_info.get('file_path', 'Unknown')}")
        print(f"  Size: {file_info.get('file_size_human', 'Unknown')}")
        print(f"  Format: {file_info.get('file_extension', 'Unknown')}")
        print(f"  Modified: {file_info.get('file_modified', 'Unknown')}")
        print(f"  Analysis: {file_info.get('analysis_timestamp', 'Unknown')}")

    except FileNotFoundError:
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Unsupported file format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
