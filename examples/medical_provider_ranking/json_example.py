"""
Example: Using provider ranking module with JSON file data.

This shows how to use the ranking module with provider data
loaded from JSON files.
"""

import json
import logging
from pathlib import Path

from da3_obsidian.medical_provider_ranking import ProviderInfo, ProviderRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_provider_candidates(json_file_path: str):
    """Load provider candidates from JSON file."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("providers", [])
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise


def load_query_provider(json_file_path: str) -> ProviderInfo:
    """Load query provider info from JSON file."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ProviderInfo(**data)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise


def main():
    """Example of using ranking module with JSON files."""

    # Example: Load query provider from JSON
    # In practice, this would come from OCR extraction
    query_data = {
        "provider_legal_name": "Metropolitan Hospital Inc",
        "provider_localised_name": "Metro Hospital",
        "phone_number": "+1-555-0123",
        "website": "metro-hospital.com",
        "street": "123 Health Street",
        "zip_code": "10001",
        "city_name": "New York",
        "province": "NY",
        "country": "United States",
    }
    provider_info = ProviderInfo(**query_data)

    # Load provider candidates from JSON file
    # This could come from a database export, API cache, etc.
    current_dir = Path(__file__).parent
    candidates_file = current_dir / "sample_providers.json"

    try:
        provider_candidates = load_provider_candidates(candidates_file)
        logger.info(f"Loaded {len(provider_candidates)} provider candidates")
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to sample data if file doesn't exist
        logger.warning("Using fallback sample data")
        provider_candidates = [
            {
                "provider_id": "PROV001",
                "name": "Metropolitan Hospital",
                "provider_legal_name": "Metropolitan Hospital Inc",
                "phone_number": "555-0123",
                "website": "metro-hospital.com",
                "zip_code": "10001",
                "city_name": "New York",
                "province": "NY",
                "country": "United States",
            },
            {
                "provider_id": "PROV002",
                "name": "Metro Medical Center",
                "provider_legal_name": "Metro Medical Center LLC",
                "phone_number": "555-0456",
                "zip_code": "10002",
                "city_name": "New York",
                "province": "NY",
                "country": "United States",
            },
        ]

    # Initialize ranker
    ranker = ProviderRanker()

    # Rank providers
    results = ranker.rank_providers(provider_info, provider_candidates, top_n=5)

    # Display results
    print(f"Found {results.total_candidates} candidates")
    print(f"Top {len(results.matches)} matches:")

    for i, match in enumerate(results.matches, 1):
        print(f"\n{i}. {match.name} (Score: {match.similarity_score:.3f})")
        print(f"   Provider ID: {match.provider_id}")
        print(f"   Legal Name: {match.provider_legal_name}")
        print(f"   Best field matches:")
        for field, score in match.match_details.items():
            if score > 0.5:  # Only show good matches
                print(f"     {field}: {score:.3f}")


if __name__ == "__main__":
    main()
