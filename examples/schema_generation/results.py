# Auto-generated from StructuredSchemaResponse — DO NOT EDIT
# Generated at: 2025-08-12 16:15:11 UTC

from __future__ import annotations

from typing import Any, Dict, Optional, List, Union, Literal
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict


class DelayRule(BaseModel):
    """Policy delay rule specifying the minimum delay for auto-payout eligibility."""

    # Model configuration: ignore unknown fields; allow population by field name
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    threshold_minutes: int = Field(
        ..., description="Minimum delay in minutes required for auto-payout eligibility.", ge=0
    )


class EligibilityResult(BaseModel):
    """Eligibility result for the travel delay claim."""

    # Model configuration: ignore unknown fields; allow population by field name
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    eligible_for_auto_payout: str = Field(
        ...,
        description="Indicates if the claim is eligible for auto-payout ('yes' or 'no').",
        min_length=2,
        max_length=3,
    )
    delay_minutes: int = Field(..., description="Total delay in minutes between scheduled and actual arrival.", ge=0)
    suggested_payout_euros: float = Field(
        ..., description="Suggested payout amount in euros based on the delay and policy.", ge=0.0
    )
    reason: str = Field(
        ...,
        description="Explanation for eligibility decision, including delay and threshold details.",
        min_length=5,
        max_length=200,
    )


class TravelDelayClaimInput(BaseModel):
    """Input data for the travel delay claim."""

    # Model configuration: ignore unknown fields; allow population by field name
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    flight_departure_datetime: str = Field(
        ...,
        description="Scheduled flight departure datetime in ISO-8601 format.",
        pattern="^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}(:\\d{2})?(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})?$",
    )
    flight_arrival_datetime: str = Field(
        ...,
        description="Scheduled flight arrival datetime in ISO-8601 format.",
        pattern="^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}(:\\d{2})?(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})?$",
    )
    actual_arrival_datetime: str = Field(
        ...,
        description="Actual flight arrival datetime in ISO-8601 format.",
        pattern="^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}(:\\d{2})?(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})?$",
    )
    delay_rule: DelayRule = Field(
        ..., description="Policy delay rule specifying the minimum delay for auto-payout eligibility."
    )
    customer_policy_id: Optional[str] = Field(
        None, description="Optional customer policy ID for confidence scoring or tracking."
    )


class TravelDelayClaimMaster(BaseModel):
    """Master aggregation class for travel delay claim processing, combining input data and eligibility result."""

    # Model configuration: ignore unknown fields; allow population by field name
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    input_data: TravelDelayClaimInput = Field(..., description="Input data for the travel delay claim.")
    eligibility_result: EligibilityResult = Field(..., description="Eligibility result for the travel delay claim.")
