"""
Advanced text extraction example showing different configuration options.
"""

import logging

from da3_obsidian.text_extraction import ImageProcessor, PDFProcessor, TextExtractor

logging.basicConfig(level=logging.INFO)


def main():
    print("=== Advanced Text Extraction Example ===\n")

    # 1. Multi-language extraction
    print("1. Multi-language Support")
    print("-" * 26)

    languages = ["en", "es", "fr", "it"]

    for lang in languages:
        extractor = TextExtractor(language=lang, gibberish_threshold=0.6)
        print(f"  - {lang.upper()}: {extractor.get_current_config()}")

    # 2. Quality assessment
    print("\n2. Quality Assessment")
    print("-" * 20)

    extractor = TextExtractor(language="en")

    # Simulate different quality texts
    high_quality = "This is a well-formatted document with clear text and proper spelling."
    low_quality = "Th1s 1s p00r qu4l1ty t3xt w1th m4ny 3rr0rs 4nd g1bb3r1sh w0rds."

    for text_type, text in [("High Quality", high_quality), ("Low Quality", low_quality)]:
        quality_scores = extractor.get_text_quality_score(text)
        is_good = extractor.is_text_high_quality(text)
        print(f"  {text_type}:")
        print(f"    - Gibberish ratios: {quality_scores}")
        print(f"    - Is high quality: {is_good}")

    # 3. Component-level usage
    print("\n3. Component-Level Usage")
    print("-" * 25)

    # Direct PDF processing
    pdf_processor = PDFProcessor(language="en", gibberish_threshold=0.7)
    print(f"  - PDF Processor configured for: {pdf_processor.language}")

    # Direct image processing
    image_processor = ImageProcessor(language="en")
    print(f"  - Image Processor configured for: {image_processor.language}")
    print(f"  - Tesseract language: {image_processor.tesseract_lang}")


if __name__ == "__main__":
    main()
