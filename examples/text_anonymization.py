"""
Advanced anonymization example with custom operators and multi-language support.
"""

import logging

from da3_obsidian.text_anonymization import TextAnonymizer

logging.basicConfig(level=logging.INFO)


def main():
    print("=== Advanced Text Anonymization Example ===\n")

    # 1. Custom anonymization operators
    print("1. Custom Anonymization Operators")
    print("-" * 35)

    anonymizer = TextAnonymizer(language="en")

    text = "Contact John Smith at john.smith@company.com or call (555) 123-4567"

    # Create custom configuration
    custom_config = anonymizer.create_custom_operators_config(
        {
            "PERSON": {"type": "replace", "new_value": "[REDACTED_PERSON]"},
            "EMAIL_ADDRESS": {"type": "mask", "masking_char": "*", "chars_to_mask": 5, "from_end": True},
            "PHONE_NUMBER": {"type": "replace", "new_value": "XXX-XXX-XXXX"},
        }
    )

    result = anonymizer.anonymize(text, anonymization_config=custom_config)
    print(f"Original: {text}")
    print(f"Custom anonymized: {result.anonymized_text}")

    # 2. Multi-language anonymization
    print("\n2. Multi-language Anonymization")
    print("-" * 32)

    test_texts = {
        "en": "My name is John Doe, email: john@example.com, phone: 555-1234",
        "es": "Mi nombre es María García, correo: maria@ejemplo.com, teléfono: 555-1234",
        "fr": "Je m'appelle Jean Dupont, email: jean@exemple.fr, téléphone: 555-1234",
        "it": "Il mio nome è Marco Rossi, email: marco@esempio.it, telefono: 555-1234",
    }

    for lang, text in test_texts.items():
        anonymizer.set_language(lang)
        result = anonymizer.anonymize(text)
        print(f"  {lang.upper()}: {result.anonymized_text}")

    # 3. Keyword-based preprocessing
    print("\n3. Keyword-based Preprocessing")
    print("-" * 31)

    anonymizer = TextAnonymizer(language="en")

    keyword_text = "Name: Alice Johnson, Phone: 555-9876, Email: alice@company.com"

    # Without keyword preprocessing
    result1 = anonymizer.anonymize(keyword_text, keyword_anonymization=False)
    print(f"Without keywords: {result1.anonymized_text}")

    # With keyword preprocessing
    result2 = anonymizer.anonymize(keyword_text, keyword_anonymization=True)
    print(f"With keywords: {result2.anonymized_text}")

    # 4. Entity-specific processing
    print("\n4. Entity-Specific Processing")
    print("-" * 29)

    # Only process specific entity types
    specific_entities = ["EMAIL_ADDRESS", "PHONE_NUMBER"]
    result = anonymizer.anonymize(text, entities=specific_entities)
    print(f"Only email/phone: {result.anonymized_text}")

    # Show all supported entities
    supported = anonymizer.get_supported_entities()
    print(f"Total supported entities: {len(supported)}")
    print(f"Examples: {supported[:10]}...")


if __name__ == "__main__":
    main()
