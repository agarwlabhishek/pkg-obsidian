import re
import logging
import unicodedata
import numpy as np
from typing import Optional, Dict, Any, Tuple, Set, List
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from .models import ProviderInfo

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Handles all similarity calculations between providers with TF-IDF enhancement."""

    def __init__(self):
        self._common_terms_cache = None
        self._common_terms_threshold = 0.05  # Terms appearing in >5% of provider names
        self._tfidf_vectorizer = None
        self._provider_corpus = None
        self._token_idf_scores = {}

    def _analyze_common_terms(self, provider_candidates: List[Dict[str, Any]]) -> Set[str]:
        """Dynamically analyze provider names to find common terms."""
        if not provider_candidates:
            return set()

        # Collect all name-based text from candidates
        all_text = []
        name_fields = [
            "name",
            "provider_legal_name",
            "provider_localised_name",
            "provider_network",
            "provider_network_name",
            "category",
        ]

        for candidate in provider_candidates:
            for field in name_fields:
                if candidate.get(field):
                    all_text.append(str(candidate[field]))

        if not all_text:
            return set()

        # Tokenize and clean all text
        all_tokens = []
        for text in all_text:
            cleaned = self._clean_text(text)
            tokens = cleaned.split()
            # Filter out very short tokens and numbers
            valid_tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
            all_tokens.extend(valid_tokens)

        if not all_tokens:
            return set()

        # Count token frequencies
        token_counts = Counter(all_tokens)
        total_providers = len(provider_candidates)

        # Find terms that appear in more than threshold% of providers
        common_terms = set()
        min_occurrences = max(1, int(total_providers * self._common_terms_threshold))

        for token, count in token_counts.items():
            if count >= min_occurrences:
                common_terms.add(token)

        logger.debug(f"Found {len(common_terms)} common terms from {len(all_tokens)} total tokens")
        logger.debug(f"Common terms: {sorted(common_terms)}")

        return common_terms

    def _build_tfidf_model(self, provider_candidates: List[Dict[str, Any]]) -> None:
        """Build TF-IDF model from provider names for token-level weighting."""
        if not provider_candidates:
            return

        # Collect all provider name variants for TF-IDF corpus
        corpus = []
        name_fields = [
            "name",
            "provider_legal_name",
            "provider_localised_name",
            "provider_network",
            "provider_network_name",
        ]

        for candidate in provider_candidates:
            candidate_names = []
            for field in name_fields:
                if candidate.get(field):
                    cleaned = self._clean_text(str(candidate[field]))
                    if cleaned:
                        candidate_names.append(cleaned)

            # Combine all name variants for this provider into one document
            if candidate_names:
                corpus.append(" ".join(candidate_names))

        if not corpus:
            return

        # Initialize TF-IDF vectorizer with token-level analysis
        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=1,  # Keep all terms (even if they appear only once)
            max_df=0.8,  # Remove terms that appear in >80% of documents
            lowercase=True,
            token_pattern=r"\b\w+\b",
        )

        # Fit the vectorizer and compute TF-IDF matrix
        try:
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
            feature_names = self._tfidf_vectorizer.get_feature_names_out()

            # Extract IDF scores for individual tokens
            idf_scores = self._tfidf_vectorizer.idf_
            self._token_idf_scores = dict(zip(feature_names, idf_scores))

            logger.debug(f"Built TF-IDF model with {len(feature_names)} features")

            # Log some high IDF (rare) terms for debugging
            sorted_terms = sorted(self._token_idf_scores.items(), key=lambda x: x[1], reverse=True)
            top_rare_terms = sorted_terms[:10]
            logger.debug(f"Top rare terms (high IDF): {top_rare_terms}")

        except Exception as e:
            logger.warning(f"Failed to build TF-IDF model: {e}")
            self._tfidf_vectorizer = None
            self._token_idf_scores = {}

    def set_provider_candidates(self, provider_candidates: List[Dict[str, Any]]) -> None:
        """Analyze provider candidates to build common terms cache and TF-IDF model."""
        self._common_terms_cache = self._analyze_common_terms(provider_candidates)
        self._build_tfidf_model(provider_candidates)
        self._provider_corpus = provider_candidates

    def _get_token_weight(self, token: str) -> float:
        """Get TF-IDF weight for a token, defaulting to 1.0 if not found."""
        if not self._token_idf_scores:
            return 1.0

        # Try exact match first
        if token in self._token_idf_scores:
            return self._token_idf_scores[token]

        # Try lowercase version
        token_lower = token.lower()
        if token_lower in self._token_idf_scores:
            return self._token_idf_scores[token_lower]

        # Default weight for unknown tokens (treat as rare)
        return max(self._token_idf_scores.values()) if self._token_idf_scores else 1.0

    def _tfidf_token_similarity(self, query_text: str, candidate_text: str) -> float:
        """Calculate similarity using TF-IDF weighted token matching."""
        if not query_text or not candidate_text or not self._token_idf_scores:
            return 0.0

        # Clean and tokenize both texts
        query_clean = self._clean_text(query_text)
        candidate_clean = self._clean_text(candidate_text)

        if not query_clean or not candidate_clean:
            return 0.0

        query_tokens = set(query_clean.split())
        candidate_tokens = set(candidate_clean.split())

        # Find matching tokens
        matching_tokens = query_tokens.intersection(candidate_tokens)

        if not matching_tokens:
            return 0.0

        # Calculate weighted similarity based on TF-IDF scores
        total_weight = 0.0
        matching_weight = 0.0

        # Weight all query tokens
        for token in query_tokens:
            weight = self._get_token_weight(token)
            total_weight += weight
            if token in matching_tokens:
                matching_weight += weight

        # Add weights for candidate tokens not in query (penalty for extra terms)
        extra_candidate_tokens = candidate_tokens - query_tokens
        for token in extra_candidate_tokens:
            weight = self._get_token_weight(token)
            total_weight += weight * 0.3  # Reduced weight for extra terms

        if total_weight == 0:
            return 0.0

        # Calculate weighted similarity score
        base_score = matching_weight / total_weight

        # Boost score for rare token matches
        rare_boost = 1.0
        for token in matching_tokens:
            token_weight = self._get_token_weight(token)
            # High IDF (rare terms) get additional boost
            if token_weight > np.percentile(list(self._token_idf_scores.values()), 75):
                rare_boost *= 1.3  # 30% boost for each rare term match

        final_score = min(1.0, base_score * rare_boost)

        # Log rare token matches for debugging
        if final_score > 0.8:
            rare_tokens = [
                token
                for token in matching_tokens
                if self._get_token_weight(token) > np.percentile(list(self._token_idf_scores.values()), 75)
            ]
            if rare_tokens:
                logger.debug(f"High similarity match with rare tokens: {rare_tokens}, score: {final_score:.3f}")

        return final_score

    def _remove_common_terms(self, text: str) -> str:
        """Remove dynamically identified common terms to focus on distinctive identifiers."""
        if not text or not self._common_terms_cache:
            return self._clean_text(text)

        # Clean and tokenize
        cleaned = self._clean_text(text)
        tokens = cleaned.split()

        # Remove common terms and very short tokens
        distinctive_tokens = [token for token in tokens if token not in self._common_terms_cache and len(token) > 1]

        return " ".join(distinctive_tokens)

    def smart_name_similarity(self, query: ProviderInfo, candidate: Dict[str, Any]) -> float:
        """Calculate smart cross-field name similarity using TF-IDF weighting."""
        # Collect all query name variants
        query_names = []
        if query.provider_legal_name:
            query_names.append(query.provider_legal_name)
        if query.provider_localised_name:
            query_names.append(query.provider_localised_name)
        if query.provider_network_name:
            query_names.append(query.provider_network_name)

        # Collect all candidate name variants
        candidate_names = []
        if candidate.get("name"):
            candidate_names.append(candidate["name"])
        if candidate.get("provider_legal_name"):
            candidate_names.append(candidate["provider_legal_name"])
        if candidate.get("provider_localised_name"):
            candidate_names.append(candidate["provider_localised_name"])
        if candidate.get("provider_network"):
            candidate_names.append(candidate["provider_network"])
        if candidate.get("provider_network_name"):
            candidate_names.append(candidate["provider_network_name"])

        # Add category-enhanced names
        if candidate.get("name") and candidate.get("category"):
            candidate_names.append(f"{candidate['category']} {candidate['name']}")
            candidate_names.append(f"{candidate['name']} {candidate['category']}")

        if not query_names or not candidate_names:
            return 0.0

        best_score = 0.0
        best_combination = None

        # Cross-product matching with TF-IDF weighting
        for query_name in query_names:
            for candidate_name in candidate_names:
                if not query_name or not candidate_name:
                    continue

                # TF-IDF weighted token similarity (primary method)
                tfidf_score = self._tfidf_token_similarity(query_name, candidate_name)

                # Traditional fuzzy matching (fallback)
                query_clean = self._clean_text(query_name)
                candidate_clean = self._clean_text(candidate_name)
                fuzzy_score = fuzz.token_sort_ratio(query_clean, candidate_clean) / 100.0

                # Distinctive terms matching (existing method)
                query_distinctive = self._remove_common_terms(query_name)
                candidate_distinctive = self._remove_common_terms(candidate_name)
                distinctive_score = 0.0
                if query_distinctive and candidate_distinctive:
                    distinctive_score = fuzz.token_sort_ratio(query_distinctive, candidate_distinctive) / 100.0

                # Combine scores with TF-IDF as primary method
                if tfidf_score > 0:
                    # TF-IDF is primary, use fuzzy as secondary
                    combined_score = tfidf_score * 0.7 + max(fuzzy_score, distinctive_score) * 0.3
                else:
                    # Fallback to traditional methods if TF-IDF unavailable
                    combined_score = max(fuzzy_score, distinctive_score)

                if combined_score > best_score:
                    best_score = combined_score
                    best_combination = (query_name, candidate_name)

        # Log best match for debugging
        if best_combination and best_score > 0.7:
            logger.debug(f"Best name match: '{best_combination[0]}' <-> '{best_combination[1]}' = {best_score:.3f}")

        return best_score

    def phone_similarity(self, query_phone: Optional[str], candidate_phone: Optional[str]) -> float:
        """Calculate phone number similarity."""
        if not query_phone or not candidate_phone:
            return 0.0

        # Extract digits only
        query_digits = re.sub(r"\D", "", query_phone)
        candidate_digits = re.sub(r"\D", "", candidate_phone)

        if not query_digits or not candidate_digits:
            return 0.0

        # Handle country codes - compare last 7-10 digits
        min_len = min(len(query_digits), len(candidate_digits))
        if min_len >= 7:
            query_suffix = query_digits[-min_len:]
            candidate_suffix = candidate_digits[-min_len:]
            return 1.0 if query_suffix == candidate_suffix else 0.0

        return 0.0

    def website_similarity(self, query_website: Optional[str], candidate_website: Optional[str]) -> float:
        """Calculate website similarity."""
        if not query_website or not candidate_website:
            return 0.0

        # Extract domain parts
        query_domain = self._extract_domain(query_website)
        candidate_domain = self._extract_domain(candidate_website)

        if query_domain and candidate_domain:
            return fuzz.ratio(query_domain, candidate_domain) / 100.0

        return 0.0

    def zip_code_similarity(self, query_zip: Optional[str], candidate_zip: Optional[str]) -> float:
        """Calculate zip code similarity."""
        if not query_zip or not candidate_zip:
            return 0.0

        # Clean zip codes (remove spaces, dashes)
        query_clean = re.sub(r"[\s\-]", "", query_zip.strip())
        candidate_clean = re.sub(r"[\s\-]", "", candidate_zip.strip())

        if not query_clean or not candidate_clean:
            return 0.0

        # Exact match
        if query_clean.lower() == candidate_clean.lower():
            return 1.0

        # Partial match for extended zip codes (e.g., 12345 vs 12345-6789)
        if len(query_clean) >= 5 and len(candidate_clean) >= 5:
            if query_clean[:5] == candidate_clean[:5]:
                return 0.8

        return 0.0

    def address_similarity(self, query: ProviderInfo, candidate: Dict[str, Any]) -> float:
        """Calculate address similarity (lower weight due to OCR issues)."""
        # Combine address components
        query_address_parts = [query.building, query.street, query.city_name, query.province]
        candidate_address_parts = [
            candidate.get("building"),
            candidate.get("street"),
            candidate.get("city_name"),
            candidate.get("province"),
        ]

        query_address = " ".join(filter(None, query_address_parts))
        candidate_address = " ".join(filter(None, candidate_address_parts))

        if not query_address or not candidate_address:
            return 0.0

        query_clean = self._clean_text(query_address)
        candidate_clean = self._clean_text(candidate_address)

        return fuzz.token_sort_ratio(query_clean, candidate_clean) / 100.0

    def tax_registration_similarity(self, query: ProviderInfo, candidate: Dict[str, Any]) -> float:
        """Calculate tax/registration number similarity."""
        # Check tax number
        if query.tax_number and candidate.get("tax_number"):
            query_tax = self._clean_text(query.tax_number)
            candidate_tax = self._clean_text(candidate.get("tax_number"))
            if query_tax == candidate_tax:
                return 1.0

        # Check registration ID
        if query.provider_registration_id and candidate.get("provider_registration_id"):
            query_reg = self._clean_text(query.provider_registration_id)
            candidate_reg = self._clean_text(candidate.get("provider_registration_id"))
            if query_reg == candidate_reg:
                return 1.0

        return 0.0

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        if not text:
            return ""

        # Convert to lowercase first
        text = text.lower()

        # Handle common accented characters for better OCR tolerance
        # Normalize unicode characters and remove accents
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        # Remove special characters and normalize spaces
        cleaned = re.sub(r"[^\w\s]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""

        # Remove protocol and www
        domain = re.sub(r"^https?://", "", url.lower())
        domain = re.sub(r"^www\.", "", domain)
        # Take everything before first slash
        domain = domain.split("/")[0]
        return domain
