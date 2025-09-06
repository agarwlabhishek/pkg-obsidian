"""
Main text anonymization interface using Presidio for PII detection and removal.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from presidio_anonymizer.entities import OperatorConfig

from .engines import (
    get_analyzer_engine,
    get_anonymizer_engine,
    get_supported_languages,
    initialize_language,
    is_language_initialized,
    validate_language_support,
)

logger = logging.getLogger(__name__)

# Comprehensive list of supported PII entities
DEFAULT_PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "LOCATION",
    "PHONE_NUMBER",
    "MEDICAL_LICENSE",
    "URL",
    "NRP",
    "CRYPTO",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_SSN",
    "UK_NHS",
    "UK_NINO",
    "ES_NIF",
    "ES_NIE",
    "IT_FISCAL_CODE",
    "IT_DRIVER_LICENSE",
    "IT_VAT_CODE",
    "IT_PASSPORT",
    "IT_IDENTITY_CARD",
    "PL_PESEL",
    "SG_NRIC_FIN",
    "SG_UEN",
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    "IN_AADHAAR",
    "IN_PAN",
    "IN_PASSPORT",
    "IN_DRIVING_LICENSE",
    "IN_VOTER",
    "FI_PERSONAL_IDENTITY_CODE",
    "PERSON",
]


@dataclass
class PIIEntity:
    """Represents a detected PII entity."""

    entity_type: str
    text: str
    start: int
    end: int
    confidence: float


@dataclass
class AnonymizationResult:
    """Result of text anonymization operation."""

    anonymized_text: str
    entities_found: List[PIIEntity]
    original_length: int
    anonymized_length: int


class TextAnonymizer:
    """
    Main interface for text anonymization using Presidio.

    Provides PII detection and anonymization capabilities with support for
    multiple languages and customizable entity types.
    """

    def __init__(self, language: str = "en", default_anonymization_operator: str = "replace"):
        """
        Initialize text anonymizer.

        Args:
            language: Language code for processing ('en', 'es', 'fr', 'it')
            default_anonymization_operator: Default operator for anonymization
        """
        self.language = language
        self.default_operator = default_anonymization_operator

        # Validate language support
        validate_language_support(language)

        # Initialize the language engine (lazy loaded)
        if not is_language_initialized(language):
            initialize_language(language)

        # Initialize keyword patterns for different languages
        self._keyword_patterns = self._initialize_keyword_patterns()

        logger.info(f"TextAnonymizer initialized for language: {language}")

    def _initialize_keyword_patterns(self) -> Dict[str, str]:
        """Initialize keyword-based anonymization patterns."""
        # Mapping of lowercase keywords to PII tags
        keyword_mapping = {
            # English
            "mr": "PERSON",
            "mrs": "PERSON",
            "miss": "PERSON",
            "ms": "PERSON",
            "dr": "PERSON",
            "prof": "PERSON",
            "sir": "PERSON",
            "madam": "PERSON",
            "name": "PERSON",
            "surname": "PERSON",
            "first name": "PERSON",
            "last name": "PERSON",
            "full name": "PERSON",
            "nickname": "PERSON",
            "phone": "PHONE_NUMBER",
            "telephone": "PHONE_NUMBER",
            "mobile": "PHONE_NUMBER",
            "email": "EMAIL_ADDRESS",
            "e-mail": "EMAIL_ADDRESS",
            "mail": "EMAIL_ADDRESS",
            "address": "LOCATION",
            "city": "LOCATION",
            "postal code": "LOCATION",
            # Spanish
            "sr": "PERSON",
            "sra": "PERSON",
            "señor": "PERSON",
            "señora": "PERSON",
            "nombre": "PERSON",
            "apellido": "PERSON",
            "nombre completo": "PERSON",
            "teléfono": "PHONE_NUMBER",
            "móvil": "PHONE_NUMBER",
            "celular": "PHONE_NUMBER",
            "correo": "EMAIL_ADDRESS",
            "correo electrónico": "EMAIL_ADDRESS",
            "dirección": "LOCATION",
            "ciudad": "LOCATION",
            "código postal": "LOCATION",
            # French
            "m.": "PERSON",
            "mme": "PERSON",
            "madame": "PERSON",
            "monsieur": "PERSON",
            "prénom": "PERSON",
            "nom": "PERSON",
            "nom complet": "PERSON",
            "téléphone": "PHONE_NUMBER",
            "portable": "PHONE_NUMBER",
            "courrier": "EMAIL_ADDRESS",
            "adresse": "LOCATION",
            "ville": "LOCATION",
            # Italian
            "sig.": "PERSON",
            "sig.ra": "PERSON",
            "signore": "PERSON",
            "signora": "PERSON",
            "nome": "PERSON",
            "cognome": "PERSON",
            "nome completo": "PERSON",
            "telefono": "PHONE_NUMBER",
            "cellulare": "PHONE_NUMBER",
            "indirizzo": "LOCATION",
            "città": "LOCATION",
        }

        # Create regex pattern
        keywords = sorted(keyword_mapping.keys(), key=len, reverse=True)
        pattern = r"(?i)\b(" + "|".join(re.escape(k) for k in keywords) + r")\b[^\w]*((?:[\w-]+[^\w]+){0,3})"

        return {"pattern": pattern, "mapping": keyword_mapping}

    def _apply_keyword_anonymization(self, text: str) -> str:
        """Apply keyword-based anonymization as preprocessing step."""

        def replace_match(match):
            keyword = match.group(1).lower()
            tag = self._keyword_patterns["mapping"].get(keyword, "PII")
            return match.group(1).capitalize() + f" <{tag}>"

        anonymized_text = re.sub(self._keyword_patterns["pattern"], replace_match, text, flags=re.IGNORECASE)

        return anonymized_text

    def analyze(self, text: str, entities: Optional[List[str]] = None) -> List[PIIEntity]:
        """
        Analyze text to detect PII entities.

        Args:
            text: Text to analyze
            entities: List of entity types to detect (uses default if None)

        Returns:
            List of detected PII entities

        Raises:
            ValueError: If invalid entities are specified
        """
        if not text or not text.strip():
            return []

        # Use default entities if none specified
        target_entities = entities or DEFAULT_PII_ENTITIES

        # Validate entities
        self._validate_entities(target_entities)

        try:
            # Get analyzer engine for the current language
            analyzer_engine = get_analyzer_engine(self.language)

            logger.debug(f"Analyzing text with entities: {target_entities}")

            results = analyzer_engine.analyze(text=text, entities=target_entities, language=self.language)

            # Convert to our dataclass format
            pii_entities = [
                PIIEntity(
                    entity_type=result.entity_type,
                    text=text[result.start : result.end],
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                )
                for result in results
            ]

            logger.debug(f"Found {len(pii_entities)} PII entities")
            return pii_entities

        except Exception as e:
            logger.error(f"Error during PII analysis: {e}")
            raise

    def anonymize(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        anonymization_config: Optional[Dict[str, OperatorConfig]] = None,
        keyword_anonymization: bool = False,
    ) -> AnonymizationResult:
        """
        Anonymize PII in text.

        Args:
            text: Text to anonymize
            entities: List of entity types to anonymize (uses default if None)
            anonymization_config: Custom anonymization operators per entity type
            keyword_anonymization: Whether to apply keyword-based preprocessing

        Returns:
            AnonymizationResult with anonymized text and metadata

        Raises:
            ValueError: If invalid configuration is provided
        """
        if not text or not text.strip():
            return AnonymizationResult(
                anonymized_text=text, entities_found=[], original_length=len(text), anonymized_length=len(text)
            )

        original_text = text
        processed_text = text

        # Apply keyword-based anonymization if requested
        if keyword_anonymization:
            logger.debug("Applying keyword-based anonymization")
            processed_text = self._apply_keyword_anonymization(processed_text)

        # Use default entities if none specified
        target_entities = entities or DEFAULT_PII_ENTITIES

        try:
            # Analyze for PII using the language-specific analyzer
            analyzer_engine = get_analyzer_engine(self.language)
            analyzer_results = analyzer_engine.analyze(
                text=processed_text, entities=target_entities, language=self.language
            )

            # Anonymize detected PII - get anonymizer engine directly
            anonymizer_engine = get_anonymizer_engine()

            # Log a debug message to verify we have a valid anonymizer
            logger.debug(f"Using anonymizer engine: {anonymizer_engine}")

            anonymization_result = anonymizer_engine.anonymize(
                text=processed_text, analyzer_results=analyzer_results, operators=anonymization_config
            )

            # Convert analyzer results to our format
            entities_found = [
                PIIEntity(
                    entity_type=result.entity_type,
                    text=processed_text[result.start : result.end],
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                )
                for result in analyzer_results
            ]

            result = AnonymizationResult(
                anonymized_text=anonymization_result.text,
                entities_found=entities_found,
                original_length=len(original_text),
                anonymized_length=len(anonymization_result.text),
            )

            logger.info(f"Anonymization completed: {len(entities_found)} entities processed")
            return result

        except Exception as e:
            logger.error(f"Error during anonymization: {e}")
            raise

    def _validate_entities(self, entities: List[str]) -> None:
        """Validate that requested entities are supported."""
        invalid_entities = [e for e in entities if e not in DEFAULT_PII_ENTITIES]
        if invalid_entities:
            raise ValueError(f"Invalid entities: {invalid_entities}. " f"Supported entities: {DEFAULT_PII_ENTITIES}")

    def get_supported_entities(self) -> List[str]:
        """Get list of supported PII entity types."""
        return DEFAULT_PII_ENTITIES.copy()

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return get_supported_languages()

    def set_language(self, language: str) -> None:
        """
        Change the processing language.

        Args:
            language: New language code

        Raises:
            ValueError: If language is not supported
        """
        validate_language_support(language)
        self.language = language

        # Ensure the language engine is initialized
        if not is_language_initialized(language):
            initialize_language(language)

        logger.info(f"Language changed to: {language}")

    def create_custom_operators_config(self, entity_configs: Dict[str, Dict[str, Any]]) -> Dict[str, OperatorConfig]:
        """
        Create custom anonymization operators configuration.

        Args:
            entity_configs: Dict mapping entity types to operator configurations

        Returns:
            Dict suitable for use with anonymize method

        Example:
            config = anonymizer.create_custom_operators_config({
                "PERSON": {"type": "replace", "new_value": "[REDACTED_PERSON]"},
                "EMAIL_ADDRESS": {"type": "mask", "masking_char": "*", "chars_to_mask": 5}
            })
        """
        operators_config = {}

        for entity_type, config in entity_configs.items():
            # Create a copy of the config to avoid modifying the original
            operator_config = config.copy()

            # Extract operator name (convert "type" to "operator_name" if present)
            if "type" in operator_config:
                operator_name = operator_config.pop("type")
            elif "operator_name" in operator_config:
                operator_name = operator_config.pop("operator_name")
            else:
                operator_name = "replace"  # default operator

            # All remaining parameters go into the params dict
            params = operator_config

            # Create OperatorConfig with correct structure
            operators_config[entity_type] = OperatorConfig(operator_name=operator_name, params=params)

        return operators_config

    def get_anonymization_summary(self, result: AnonymizationResult) -> Dict[str, Any]:
        """
        Get summary statistics for anonymization result.

        Args:
            result: AnonymizationResult to summarize

        Returns:
            Dictionary with summary statistics
        """
        entity_counts = {}
        for entity in result.entities_found:
            entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1

        return {
            "total_entities_found": len(result.entities_found),
            "entities_by_type": entity_counts,
            "original_length": result.original_length,
            "anonymized_length": result.anonymized_length,
            "length_change": result.anonymized_length - result.original_length,
            "average_confidence": (
                sum(e.confidence for e in result.entities_found) / len(result.entities_found)
                if result.entities_found
                else 0.0
            ),
        }
