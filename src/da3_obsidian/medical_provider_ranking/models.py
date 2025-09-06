from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """Healthcare provider and facility information."""

    # Provider identification
    provider_legal_name: Optional[str] = Field(None, description="Official registered name")
    provider_localised_name: Optional[str] = Field(None, description="Local language name")
    provider_network_name: Optional[str] = Field(None, description="Provider network name")
    provider_registration_id: Optional[str] = Field(None, description="Government registration number")
    tax_number: Optional[str] = Field(None, description="Tax identification number")

    # Contact information
    country_code_number: Optional[str] = Field(None, description="International dialing code")
    phone_number: Optional[str] = Field(None, description="Primary contact number")
    website: Optional[str] = Field(None, description="Official website URL")

    # Address components
    building: Optional[str] = Field(None, description="Building name/number")
    floor: Optional[str] = Field(None, description="Floor designation")
    house_no: Optional[str] = Field(None, description="Street number")
    street: Optional[str] = Field(None, description="Street name")
    neighborhood: Optional[str] = Field(None, description="Local area designation")
    zip_code: Optional[str] = Field(None, description="Postal code")
    city_name: Optional[str] = Field(None, description="City name")
    province: Optional[str] = Field(None, description="State/province name")
    country: Optional[str] = Field(None, description="Country name")
    region: Optional[str] = Field(None, description="Continental region")


class ProviderMatch(BaseModel):
    """Ranked provider match result."""

    provider_id: str
    name: str
    provider_legal_name: Optional[str]
    similarity_score: float
    match_details: Dict[str, float]
    candidate_data: Dict[str, Any]


class RankingResponse(BaseModel):
    """Provider ranking response."""

    total_candidates: int
    matches: List[ProviderMatch]
    query_provider: ProviderInfo


@dataclass
class SimilarityWeights:
    """Scoring weights for different matching fields."""

    name_match: float = 0.35  # Smart name matching base weight
    legal_name_match: float = 0.20  # Combined with name_match for total 55%
    phone_match: float = 0.15
    website_match: float = 0.10
    zip_code_match: float = 0.10
    address_match: float = 0.05
    tax_registration_match: float = 0.05
