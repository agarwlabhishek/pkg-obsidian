"""Utility functions for provider ranking module."""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from .models import ProviderInfo, RankingResponse

logger = logging.getLogger(__name__)


def load_provider_candidates_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load provider candidates from JSON file.

    Args:
        file_path: Path to JSON file containing provider data

    Returns:
        List of provider candidate dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys for provider lists
            for key in ["providers", "data", "candidates"]:
                if key in data:
                    return data[key]
            # If no standard key found, assume the dict is a single provider
            return [data]
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

    except FileNotFoundError:
        logger.error(f"Provider candidates file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise


def load_query_provider_from_json(file_path: str) -> ProviderInfo:
    """Load query provider info from JSON file.

    Args:
        file_path: Path to JSON file containing provider query data

    Returns:
        ProviderInfo object

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        ValueError: If data doesn't match ProviderInfo schema
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ProviderInfo(**data)

    except FileNotFoundError:
        logger.error(f"Query provider file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Invalid provider data structure in {file_path}: {e}")
        raise


def save_ranking_results_to_json(results: RankingResponse, file_path: str) -> None:
    """Save ranking results to JSON file.

    Args:
        results: RankingResponse object
        file_path: Output file path
    """
    try:
        # Convert to JSON-serializable format
        output_data = {
            "query_provider": results.query_provider.dict(),
            "total_candidates": results.total_candidates,
            "matches": [
                {
                    "provider_id": match.provider_id,
                    "name": match.name,
                    "provider_legal_name": match.provider_legal_name,
                    "similarity_score": match.similarity_score,
                    "match_details": match.match_details,
                }
                for match in results.matches
            ],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save results to {file_path}: {e}")
        raise


def validate_provider_candidate(candidate: Dict[str, Any]) -> bool:
    """Validate that a provider candidate has required fields.

    Args:
        candidate: Provider candidate dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["provider_id", "name"]

    for field in required_fields:
        if field not in candidate or not candidate[field]:
            logger.warning(f"Provider candidate missing required field '{field}': {candidate}")
            return False

    return True


def filter_valid_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter provider candidates to only include valid ones.

    Args:
        candidates: List of provider candidate dictionaries

    Returns:
        Filtered list containing only valid candidates
    """
    valid_candidates = []

    for candidate in candidates:
        if validate_provider_candidate(candidate):
            valid_candidates.append(candidate)

    logger.info(f"Filtered {len(candidates)} candidates to {len(valid_candidates)} valid candidates")
    return valid_candidates
