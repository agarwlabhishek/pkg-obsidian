import logging
from typing import Dict, Any, Tuple, List

from .models import ProviderInfo, ProviderMatch, RankingResponse, SimilarityWeights
from .similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


class ProviderRanker:
    """Medical provider ranking and similarity matching."""

    def __init__(self):
        self.similarity_calculator = SimilarityCalculator()
        self.weights = SimilarityWeights()

    def rank_providers(
        self, provider_info: ProviderInfo, provider_candidates: List[Dict[str, Any]], top_n: int = 10
    ) -> RankingResponse:
        """Rank providers by similarity to input provider info.

        Args:
            provider_info: Query provider information from OCR extraction
            provider_candidates: List of provider candidates to rank against
            top_n: Maximum number of matches to return

        Returns:
            RankingResponse with ranked matches
        """
        logger.info(f"Starting provider ranking for {provider_info.provider_legal_name or 'Unknown'}")
        logger.info(f"Evaluating {len(provider_candidates)} provider candidates")

        # Analyze provider candidates to build dynamic common terms
        self.similarity_calculator.set_provider_candidates(provider_candidates)

        # Calculate similarity scores
        scored_matches = []
        for candidate in provider_candidates:
            score, details = self._calculate_similarity(provider_info, candidate)
            if score > 0.1:  # Minimum threshold
                match = ProviderMatch(
                    provider_id=candidate["provider_id"],
                    name=candidate["name"],
                    provider_legal_name=candidate.get("provider_legal_name"),
                    similarity_score=score,
                    match_details=details,
                    candidate_data=candidate,
                )
                scored_matches.append(match)

        # Sort by score and return top N
        scored_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        top_matches = scored_matches[:top_n]

        logger.info(f"Returning {len(top_matches)} matches (threshold: 0.1+)")

        return RankingResponse(
            total_candidates=len(provider_candidates), matches=top_matches, query_provider=provider_info
        )

    def _calculate_similarity(self, query: ProviderInfo, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted similarity score between query and candidate."""
        details = {}
        total_score = 0.0

        # Smart name matching (cross-field combination)
        name_score = self.similarity_calculator.smart_name_similarity(query, candidate)
        details["name_similarity"] = name_score
        total_score += name_score * (self.weights.name_match + self.weights.legal_name_match)

        # Phone similarity
        phone_score = self.similarity_calculator.phone_similarity(query.phone_number, candidate.get("phone_number"))
        details["phone_similarity"] = phone_score
        total_score += phone_score * self.weights.phone_match

        # Website similarity
        website_score = self.similarity_calculator.website_similarity(query.website, candidate.get("website"))
        details["website_similarity"] = website_score
        total_score += website_score * self.weights.website_match

        # Zip code similarity
        zip_score = self.similarity_calculator.zip_code_similarity(query.zip_code, candidate.get("zip_code"))
        details["zip_code_similarity"] = zip_score
        total_score += zip_score * self.weights.zip_code_match

        # Address similarity
        address_score = self.similarity_calculator.address_similarity(query, candidate)
        details["address_similarity"] = address_score
        total_score += address_score * self.weights.address_match

        # Tax/registration similarity
        tax_score = self.similarity_calculator.tax_registration_similarity(query, candidate)
        details["tax_registration_similarity"] = tax_score
        total_score += tax_score * self.weights.tax_registration_match

        return total_score, details
