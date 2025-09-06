import logging

from da3_obsidian.medical_provider_ranking.models import ProviderInfo
from da3_obsidian.medical_provider_ranking.ranker import ProviderRanker

logging.basicConfig(level=logging.INFO)


def main():
    """Example usage of the provider ranking system with sample data."""
    # Example OCR-extracted provider info
    provider_info = ProviderInfo(
        provider_legal_name="Metropolitan Hospital Inc",
        provider_localised_name="Metro Hospital",
        phone_number="+1-555-0123",
        website="metro-hospital.com",
        street="123 Health Street",
        zip_code="10001",
        city_name="New York",
        province="NY",
        country="United States",
    )

    # Sample provider candidates (in practice, this comes from your data source)
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

    # Get ranked matches
    results = ranker.rank_providers(provider_info, provider_candidates, top_n=5)

    print(f"Found {results.total_candidates} candidates")
    print(f"Top {len(results.matches)} matches:")

    for i, match in enumerate(results.matches, 1):
        print(f"\n{i}. {match.name} (Score: {match.similarity_score:.3f})")
        print(f"   Provider ID: {match.provider_id}")
        print(f"   Legal Name: {match.provider_legal_name}")
        print(f"   Match Details: {match.match_details}")


if __name__ == "__main__":
    main()
