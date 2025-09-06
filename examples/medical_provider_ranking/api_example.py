"""
Example: Using provider ranking module with API data source.

This shows how to integrate the ranking module with an external API
like the GNP Data API.
"""

import re
import logging
from typing import List, Dict, Any, Optional

import requests

from da3_obsidian.medical_provider_ranking import ProviderInfo, ProviderRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApiClient:
    """Client for GNP Data API communication."""

    def __init__(self, api_base_url: str = "http://localhost:8890"):
        self.api_base_url = api_base_url

    def fetch_providers(self, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch providers from GNP API with optional country filter."""
        url = f"{self.api_base_url}/providers"
        params = {}

        if country:
            # Clean country name for API
            params["country"] = country
            logger.info(f"Filtering by country: {country}")

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("providers", [])
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def _clean_country_name(self, country: str) -> str:
        """Clean country name for API query."""
        if not country:
            return ""

        # Basic text cleaning for country names
        cleaned = re.sub(r"[^\w\s]", " ", country.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


def main():
    """Example of using ranking module with API data."""
    # Example OCR-extracted provider info
    provider_info = ProviderInfo(
        name="Clinica Rotger",
        provider_localised_name="Clinica Rotger",
        provider_network_name="quironsalud",
        country="Spain",
    )

    # Fetch provider candidates from API
    api_client = ApiClient()
    provider_candidates = api_client.fetch_providers(provider_info.country)

    # Initialize ranker (no API dependency)
    ranker = ProviderRanker()

    # Optional: Customize common terms threshold
    ranker = ProviderRanker()

    # Less aggressive common term removal (10% threshold)
    ranker.similarity_calculator._common_terms_threshold = 0.05

    # Rank providers using fetched candidates
    results = ranker.rank_providers(provider_info, provider_candidates, top_n=10)

    # Display results
    print(f"Found {results.total_candidates} candidates")
    print(f"Top {len(results.matches)} matches:")

    for i, match in enumerate(results.matches, 1):
        print(f"\n{i}. {match.name} (Score: {match.similarity_score:.3f})")
        print(f"   Provider ID: {match.provider_id}")
        print(f"   Legal Name: {match.provider_legal_name}")
        print(f"   Match Details: {match.match_details}")


if __name__ == "__main__":
    main()
