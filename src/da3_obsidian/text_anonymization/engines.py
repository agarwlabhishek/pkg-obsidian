"""
Presidio engine management for text anonymization.

This module handles the initialization and management of Presidio's
AnalyzerEngine and AnonymizerEngine with multi-language support.
"""

import logging
from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# Global engine instances
_analyzer_engines: Dict[str, AnalyzerEngine] = {}
_anonymizer_engine: Optional[AnonymizerEngine] = None
_initialized_languages: List[str] = []

# Default language models configuration
DEFAULT_MODELS_CONFIG = [
    {"lang_code": "en", "model_name": "en_core_web_lg"},
    {"lang_code": "es", "model_name": "es_core_news_lg"},
    {"lang_code": "fr", "model_name": "fr_core_news_lg"},
    {"lang_code": "it", "model_name": "it_core_news_lg"},
]


class PresidioEngineManager:
    """Manages Presidio engines with singleton pattern and lazy language loading."""

    def __init__(self, models_config: Optional[List[dict]] = None):
        """
        Initialize engine manager.

        Args:
            models_config: List of language model configurations.
                          If None, uses default configuration.
        """
        self.models_config = models_config or DEFAULT_MODELS_CONFIG
        self._initialized = False
        self._model_map = {model["lang_code"]: model for model in self.models_config}
        self._supported_languages = [model["lang_code"] for model in self.models_config]

    def initialize_language(self, language: str) -> None:
        """
        Initialize AnalyzerEngine for a specific language.

        Args:
            language: Language code to initialize

        Raises:
            ValueError: If language is not supported
            RuntimeError: If initialization fails
        """
        global _analyzer_engines, _anonymizer_engine, _initialized_languages

        # Validate language
        if language not in self._supported_languages:
            raise ValueError(
                f"Unsupported language '{language}'. " f"Supported languages: {', '.join(self._supported_languages)}"
            )

        # Check if already initialized
        if language in _initialized_languages:
            logger.debug(f"Language '{language}' already initialized")
            return

        try:
            # Initialize anonymizer engine if not already done
            if _anonymizer_engine is None:
                _anonymizer_engine = AnonymizerEngine()
                logger.info("AnonymizerEngine initialized successfully")

            # Get the specific model for this language
            lang_model = self._model_map[language]

            # Configuration for NLP Engine with just this language
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [lang_model],
            }

            logger.info(f"Initializing NLP engine for language: {language}")

            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            # Initialize Analyzer engine for this language
            _analyzer_engines[language] = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])

            # Add to initialized languages
            _initialized_languages.append(language)

            # Mark as initialized
            self._initialized = True
            logger.info(f"Presidio analyzer engine initialized for language: {language}")

        except Exception as e:
            logger.error(f"Failed to initialize Presidio engine for language {language}: {e}")
            raise RuntimeError(f"Engine initialization failed: {e}") from e

    def get_analyzer_engine(self, language: str) -> AnalyzerEngine:
        """
        Get the analyzer engine instance for a specific language.

        Args:
            language: Language to get engine for

        Returns:
            AnalyzerEngine for the specified language

        Raises:
            ValueError: If language is not supported
            RuntimeError: If initialization fails
        """
        if language not in _initialized_languages:
            self.initialize_language(language)

        return _analyzer_engines[language]

    def get_anonymizer_engine(self) -> AnonymizerEngine:
        """
        Get the anonymizer engine instance.

        Returns:
            AnonymizerEngine instance

        Raises:
            RuntimeError: If engine initialization fails
        """
        global _anonymizer_engine

        # Initialize anonymizer if not already done
        if _anonymizer_engine is None:
            _anonymizer_engine = AnonymizerEngine()
            logger.info("AnonymizerEngine initialized successfully")

        return _anonymizer_engine

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self._supported_languages.copy()

    def get_initialized_languages(self) -> List[str]:
        """Get list of languages that have been initialized."""
        return _initialized_languages.copy()

    def is_language_initialized(self, language: str) -> bool:
        """Check if a specific language is initialized."""
        return language in _initialized_languages

    def is_initialized(self) -> bool:
        """Check if at least one language engine is initialized."""
        return self._initialized

    def reset_engines(self) -> None:
        """Reset engines (mainly for testing purposes)."""
        global _analyzer_engines, _anonymizer_engine, _initialized_languages
        _analyzer_engines = {}
        _anonymizer_engine = None
        _initialized_languages = []
        self._initialized = False
        logger.info("Engines reset")


# Global manager instance
_engine_manager = PresidioEngineManager()


def get_analyzer_engine(language: str = "en") -> AnalyzerEngine:
    """
    Get the analyzer engine instance for a specific language.

    Args:
        language: Language code to get engine for (default: 'en')

    Returns:
        AnalyzerEngine instance for the specified language

    Raises:
        ValueError: If language is not supported
        RuntimeError: If engine initialization fails
    """
    return _engine_manager.get_analyzer_engine(language)


def get_anonymizer_engine() -> AnonymizerEngine:
    """
    Get the global anonymizer engine instance.

    Returns:
        AnonymizerEngine instance

    Raises:
        RuntimeError: If engine initialization fails
    """
    global _anonymizer_engine

    # If the anonymizer engine hasn't been initialized yet, initialize it
    if _anonymizer_engine is None:
        _anonymizer_engine = AnonymizerEngine()
        logger.info("AnonymizerEngine initialized directly")

    return _anonymizer_engine


def get_supported_languages() -> List[str]:
    """
    Get list of supported languages.

    Returns:
        List of language codes
    """
    return _engine_manager.get_supported_languages()


def get_initialized_languages() -> List[str]:
    """
    Get list of languages that have been initialized.

    Returns:
        List of initialized language codes
    """
    return _engine_manager.get_initialized_languages()


def is_language_initialized(language: str) -> bool:
    """
    Check if a specific language is initialized.

    Args:
        language: Language code to check

    Returns:
        True if language is initialized
    """
    return _engine_manager.is_language_initialized(language)


def is_engine_initialized() -> bool:
    """
    Check if at least one language engine is initialized.

    Returns:
        True if at least one language engine is ready
    """
    return _engine_manager.is_initialized()


def initialize_language(language: str) -> None:
    """
    Initialize engine for a specific language.

    Args:
        language: Language code to initialize

    Raises:
        ValueError: If language is not supported
        RuntimeError: If initialization fails
    """
    _engine_manager.initialize_language(language)


def initialize_engines(models_config: Optional[List[dict]] = None) -> None:
    """
    Initialize engine manager with custom configuration.

    Note: This doesn't load any language models until they're requested.

    Args:
        models_config: Custom language model configuration
    """
    global _engine_manager
    if models_config:
        _engine_manager = PresidioEngineManager(models_config)


def reset_engines() -> None:
    """Reset engines (mainly for testing)."""
    global _analyzer_engines, _anonymizer_engine, _initialized_languages
    _analyzer_engines = {}
    _anonymizer_engine = None
    _initialized_languages = []
    _engine_manager._initialized = False
    logger.info("Engines reset")


def validate_language_support(language: str) -> None:
    """
    Validate that a language is supported.

    Args:
        language: Language code to validate

    Raises:
        ValueError: If language is not supported
    """
    supported = get_supported_languages()
    if language not in supported:
        raise ValueError((f"Unsupported language '{language}'. " f"Supported languages: {', '.join(supported)}"))
