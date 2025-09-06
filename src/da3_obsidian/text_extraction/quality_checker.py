"""
Text quality validation module for OCR output assessment.
"""

import logging
from typing import Set, Tuple

import numpy as np
from spellchecker import SpellChecker

logger = logging.getLogger(__name__)

# Try to import NLTK words, fallback to empty set if not available
try:
    import importlib.util

    if importlib.util.find_spec("nltk") is not None:
        from nltk.corpus import words

        ENGLISH_WORDS = set(words.words())
    else:
        raise ImportError("NLTK not found")
except ImportError:
    logger.warning("NLTK not available, using minimal English word list")
    ENGLISH_WORDS = set()
except Exception:
    ENGLISH_WORDS = set()


class QualityChecker:
    """Validates OCR text quality using spell checking and gibberish detection."""

    def __init__(self):
        """Initialize quality checker."""
        self._spell_checkers = {}

    def replace_spanish_characters(self, text: str) -> str:
        """Replace Spanish special characters with English equivalents."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        translation_table = str.maketrans(
            {
                "á": "a",
                "é": "e",
                "í": "i",
                "ó": "o",
                "ú": "u",
                "Á": "A",
                "É": "E",
                "Í": "I",
                "Ó": "O",
                "Ú": "U",
                "ñ": "n",
                "Ñ": "N",
                "ü": "u",
                "Ü": "U",
            }
        )
        return text.translate(translation_table)

    def clean_string(self, input_text: str) -> str:
        """Clean text by removing non-alpha tokens and converting to lowercase."""
        if not isinstance(input_text, str):
            raise ValueError("Input must be a string")

        cleaned_string = " ".join([token.lower() for token in input_text.split() if token.isalpha() and len(token) > 2])
        return cleaned_string

    def _get_spell_checker(self, language: str) -> SpellChecker:
        """Get or create spell checker for language."""
        if language not in self._spell_checkers:
            try:
                spell = SpellChecker(language=language, distance=1, case_sensitive=False)
                self._spell_checkers[language] = spell
                logger.debug(f"Initialized spell checker for language: {language}")
            except Exception as e:
                logger.warning(f"Failed to initialize spell checker for {language}: {e}")
                # Fallback to English
                spell = SpellChecker(language="en", distance=1, case_sensitive=False)
                self._spell_checkers[language] = spell

        return self._spell_checkers[language]

    def spellcheck_words(self, extracted_words, language: str = "en") -> Tuple[Set[str], Set[str]]:
        """Separate known and gibberish words using spell checker."""
        try:
            spell = self._get_spell_checker(language)

            known_words = spell.known(extracted_words)
            gibberish_words = spell.unknown(extracted_words)

            logger.debug(f"Spellcheck: {len(known_words)} known, {len(gibberish_words)} unknown")
            return known_words, gibberish_words

        except Exception as e:
            logger.error(f"Error in spellcheck_words: {e}")
            # Return empty sets as fallback
            return set(), set(extracted_words)

    def calculate_gibberish_ratio(self, ocr_text: str, language: str = "en") -> Tuple[float, float]:
        """Calculate ratio of gibberish words to known words in OCR text."""
        try:
            ocr_tokens = ocr_text.split()
            if not ocr_tokens:
                return 0.0, 0.0

            # Primary spellcheck
            all_known_words, all_gibberish_words = self.spellcheck_words(ocr_tokens, language)

            # If language is not English, also try English for unknown words
            if language != "en" and all_gibberish_words:
                en_known, en_gib = self.spellcheck_words(list(all_gibberish_words), "en")
                all_known_words.update(en_known)
                all_gibberish_words = en_gib

            # Count occurrences in original text
            known_count = sum(1 for word in ocr_tokens if word in all_known_words)
            gibberish_count = sum(1 for word in ocr_tokens if word in all_gibberish_words)

            # Calculate ratios
            if len(all_known_words) == 0:
                gibberish_set_ratio = np.inf
            else:
                gibberish_set_ratio = len(all_gibberish_words) / len(all_known_words)

            if known_count == 0:
                gibberish_tokens_ratio = np.inf
            else:
                gibberish_tokens_ratio = gibberish_count / known_count

            logger.debug(f"Gibberish ratios - set: {gibberish_set_ratio:.3f}, tokens: {gibberish_tokens_ratio:.3f}")

            return gibberish_set_ratio, gibberish_tokens_ratio

        except Exception as e:
            logger.error(f"Error calculating gibberish ratio: {e}")
            return np.inf, np.inf

    def determine_confidence(
        self, digital_text: str, ocr_text: str, language: str, gibberish_threshold: float = 0.25
    ) -> str:
        """Determine confidence level based on digital vs OCR text quality."""
        try:
            # Clean both texts
            valid_digital_text = self.clean_string(digital_text)
            valid_ocr_text = self.clean_string(ocr_text)

            digital_len = len(valid_digital_text.split())
            ocr_len = len(valid_ocr_text.split())
            total_tokens = digital_len + ocr_len

            # No tokens extracted
            if total_tokens == 0:
                logger.info("Confidence: Low (no tokens)")
                return "low"

            # High digital text percentage = high confidence
            digital_percentage = (digital_len / total_tokens) * 100
            if digital_percentage > 70:
                logger.info(f"Confidence: High (digital: {digital_percentage:.1f}%)")
                return "high"

            # Check OCR quality for Spanish text
            if language == "es":
                valid_ocr_text = self.replace_spanish_characters(valid_ocr_text)

            # Calculate gibberish ratios
            gibberish_set_ratio, gibberish_tokens_ratio = self.calculate_gibberish_ratio(valid_ocr_text, language)

            min_ratio = min(gibberish_set_ratio, gibberish_tokens_ratio)
            logger.info(f"Confidence: OCR gibberish ratio: {min_ratio:.3f}")

            # Determine confidence based on gibberish threshold
            if min_ratio > gibberish_threshold:
                return "low"
            else:
                return "medium"

        except Exception as e:
            logger.error(f"Error determining confidence: {e}")
            return "low"

    def is_text_valid(self, text: str, language: str, gibberish_threshold: float) -> bool:
        """Check if text meets quality standards."""
        if not text or not text.strip():
            return False

        valid_text = self.clean_string(text)
        word_count = len(valid_text.split())

        # Short text is generally accepted
        if word_count <= 10:
            return True

        # For longer text, check gibberish ratio
        try:
            if language == "es":
                valid_text = self.replace_spanish_characters(valid_text)

            gib_set_ratio, gib_tokens_ratio = self.calculate_gibberish_ratio(valid_text, language)
            min_ratio = min(gib_set_ratio, gib_tokens_ratio)

            return min_ratio < gibberish_threshold

        except Exception as e:
            logger.error(f"Error validating text: {e}")
            return False
