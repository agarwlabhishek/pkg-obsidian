#!/usr/bin/env python3
"""
Comprehensive example demonstrating email preprocessing with text extraction and anonymization.
This script processes .msg files and produces a formatted report with all extracted information.
"""

import logging
import os
from datetime import datetime

from da3_obsidian.email_preprocessing import EmailPreprocessor
from da3_obsidian.text_anonymization import TextAnonymizer
from da3_obsidian.text_extraction import TextExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def process_email_comprehensive(msg_file_path: str) -> str:
    """
    Process an email file comprehensively and return formatted results.

    Args:
        msg_file_path: Path to the .msg file

    Returns:
        Formatted string containing all extracted and processed information
    """
    # Initialize all processors
    preprocessor = EmailPreprocessor(preserve_temp_files=False)
    extractor = TextExtractor(language="en")
    anonymizer = TextAnonymizer(language="en")

    # Build the comprehensive report
    report_parts = []

    # Header
    report_parts.append("=" * 80)
    report_parts.append("📧 COMPREHENSIVE EMAIL PROCESSING REPORT")
    report_parts.append("=" * 80)
    report_parts.append(f"⏰ Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_parts.append(f"📁 Source file: {msg_file_path}")
    report_parts.append("")

    try:
        # Process the email
        email_data = preprocessor.process_msg_file(msg_file_path)

        # 1. EMAIL METADATA SECTION
        report_parts.append("📋 EMAIL METADATA")
        report_parts.append("-" * 40)
        report_parts.append(f"Subject: {email_data.subject or 'No Subject'}")
        report_parts.append(f"From: {email_data.sender or 'Unknown Sender'}")
        report_parts.append(f"To: {email_data.to or 'No Recipients'}")
        report_parts.append(f"CC: {email_data.cc or 'None'}")
        report_parts.append(f"Date: {email_data.date or 'Unknown Date'}")
        report_parts.append(f"Message ID: {email_data.message_id or 'Not Available'}")
        report_parts.append(f"Importance: {email_data.importance or 'Normal'}")
        report_parts.append("")

        # 2. EMAIL BODY PROCESSING SECTION
        report_parts.append("📝 EMAIL BODY ANALYSIS")
        report_parts.append("-" * 40)

        if email_data.body:
            body_length = len(email_data.body)
            word_count = len(email_data.body.split())

            report_parts.append(f"Body length: {body_length:,} characters")
            report_parts.append(f"Word count: {word_count:,} words")

            # Show first 200 characters of body
            body_preview = email_data.body[:200].replace("\n", " ").replace("\r", " ")
            if len(email_data.body) > 200:
                body_preview += "..."
            report_parts.append(f"Body preview: {body_preview}")

            # Anonymize email body
            try:
                body_result = anonymizer.anonymize(email_data.body)
                report_parts.append(f"PII entities found in body: {len(body_result.entities_found)}")

                if body_result.entities_found:
                    entity_counts = {}
                    for entity in body_result.entities_found:
                        entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1

                    report_parts.append("PII entities by type:")
                    for entity_type, count in entity_counts.items():
                        report_parts.append(f"  • {entity_type}: {count}")

                # Show anonymized body preview
                anonymized_preview = body_result.anonymized_text[:200].replace("\n", " ").replace("\r", " ")
                if len(body_result.anonymized_text) > 200:
                    anonymized_preview += "..."
                report_parts.append(f"Anonymized preview: {anonymized_preview}")

            except Exception as e:
                report_parts.append(f"Body anonymization failed: {e}")
        else:
            report_parts.append("No body content found")

        report_parts.append("")

        # 3. ATTACHMENTS SECTION
        report_parts.append("📎 ATTACHMENTS ANALYSIS")
        report_parts.append("-" * 40)

        if email_data.attachments:
            report_parts.append(f"Total attachments: {len(email_data.attachments)}")

            # Get attachment statistics
            handler = preprocessor.msg_processor.attachment_handler
            stats = handler.get_attachment_stats(email_data.attachments)

            report_parts.append(f"Total size: {stats['total_size_mb']} MB ({stats['total_size_bytes']:,} bytes)")
            report_parts.append("Content types:")
            for content_type, count in stats["content_types"].items():
                report_parts.append(f"  • {content_type}: {count} files")
            report_parts.append("")

            # Process each attachment
            all_extracted_text = []
            total_extracted_chars = 0
            total_pii_entities = 0

            for i, attachment in enumerate(email_data.attachments, 1):
                report_parts.append(f"📄 Attachment {i}: {attachment.filename}")
                report_parts.append(f"   Size: {attachment.size:,} bytes")
                report_parts.append(f"   Type: {attachment.content_type}")
                report_parts.append(f"   Path: {attachment.path}")

                # Extract text based on file type
                extracted_text = ""
                extraction_method = "None"

                try:
                    filename_lower = attachment.filename.lower()

                    if filename_lower.endswith(".pdf"):
                        extraction_method = "PDF extraction"
                        text, confidence = extractor.extract_from_pdf(attachment.path)
                        extracted_text = text
                        report_parts.append(f"   PDF extraction confidence: {confidence}")

                    elif filename_lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                        extraction_method = "OCR (Image)"
                        extracted_text = extractor.extract_from_image(attachment.path)

                        # Check OCR quality
                        extractor.get_text_quality_score(extracted_text)
                        is_high_quality = extractor.is_text_high_quality(extracted_text)
                        report_parts.append(f"   OCR quality: {'High' if is_high_quality else 'Low'}")

                    elif filename_lower.endswith((".txt", ".rtf")):
                        extraction_method = "Plain text"
                        try:
                            with open(attachment.path, "r", encoding="utf-8", errors="ignore") as f:
                                extracted_text = f.read()
                        except Exception:
                            with open(attachment.path, "r", encoding="latin-1", errors="ignore") as f:
                                extracted_text = f.read()

                    # Process extracted text
                    if extracted_text:
                        char_count = len(extracted_text)
                        word_count = len(extracted_text.split())
                        total_extracted_chars += char_count

                        report_parts.append(f"   Extraction method: {extraction_method}")
                        report_parts.append(f"   Extracted: {char_count:,} characters, {word_count:,} words")

                        # Anonymize extracted text
                        try:
                            anonymization_result = anonymizer.anonymize(extracted_text)
                            pii_count = len(anonymization_result.entities_found)
                            total_pii_entities += pii_count

                            if pii_count > 0:
                                report_parts.append(f"   PII entities found: {pii_count}")

                                # Show entity types
                                entity_types = set(e.entity_type for e in anonymization_result.entities_found)
                                report_parts.append(f"   PII types: {', '.join(entity_types)}")
                            else:
                                report_parts.append("   No PII entities detected")

                            # Add to overall extracted text
                            all_extracted_text.append(f"\n--- FROM {attachment.filename} ---\n")
                            all_extracted_text.append(anonymization_result.anonymized_text)

                        except Exception as e:
                            report_parts.append(f"   Anonymization failed: {e}")
                            all_extracted_text.append(f"\n--- FROM {attachment.filename} (NOT ANONYMIZED) ---\n")
                            all_extracted_text.append(extracted_text)
                    else:
                        report_parts.append("   No text extracted")

                except Exception as e:
                    report_parts.append(f"   Processing failed: {e}")

                report_parts.append("")

            # 4. OVERALL STATISTICS
            report_parts.append("📊 OVERALL STATISTICS")
            report_parts.append("-" * 40)
            report_parts.append(f"Total characters extracted: {total_extracted_chars:,}")
            report_parts.append(f"Total PII entities found: {total_pii_entities}")
            report_parts.append(
                f"Files processed successfully: {len([a for a in email_data.attachments if os.path.exists(a.path)])}"
            )

            # Calculate email summary
            email_summary = preprocessor.get_email_summary(email_data)
            report_parts.append(f"Temporary files location: {email_summary['temp_directory']}")

            report_parts.append("")

            # 5. CONSOLIDATED EXTRACTED TEXT (if any)
            if all_extracted_text:
                report_parts.append("📄 CONSOLIDATED EXTRACTED TEXT (ANONYMIZED)")
                report_parts.append("-" * 40)

                consolidated_text = "".join(all_extracted_text)

                # Show first 500 characters as preview
                preview_text = consolidated_text[:500].replace("\n", " ").replace("\r", " ")
                if len(consolidated_text) > 500:
                    preview_text += "..."

                report_parts.append(f"Preview ({len(consolidated_text):,} total characters):")
                report_parts.append(f"{preview_text}")
                report_parts.append("")

                # Final anonymization of all text together
                try:
                    final_anonymization = anonymizer.anonymize(consolidated_text)
                    summary = anonymizer.get_anonymization_summary(final_anonymization)

                    report_parts.append("Final anonymization summary:")
                    report_parts.append(f"  Total entities: {summary['total_entities_found']}")
                    report_parts.append(f"  Average confidence: {summary['average_confidence']:.2f}")
                    report_parts.append("  Entities by type:")
                    for entity_type, count in summary["entities_by_type"].items():
                        report_parts.append(f"    • {entity_type}: {count}")

                except Exception as e:
                    report_parts.append(f"Final anonymization failed: {e}")
        else:
            report_parts.append("No attachments found")

        # Clean up temporary files
        preprocessor.cleanup_email_data(email_data)
        report_parts.append("")
        report_parts.append("✅ Temporary files cleaned up")

    except Exception as e:
        report_parts.append(f"❌ ERROR: Failed to process email: {e}")

    # Footer
    report_parts.append("")
    report_parts.append("=" * 80)
    report_parts.append("🏁 PROCESSING COMPLETE")
    report_parts.append("=" * 80)

    return "\n".join(report_parts)


def main():
    """Main function to run the comprehensive example."""
    import sys

    # Get the MSG file path
    if len(sys.argv) > 1:
        msg_file = sys.argv[1]
    else:
        msg_file = "sample_documents/email.msg"

    # Check if file exists
    if not os.path.exists(msg_file):
        print(f"ERROR: File not found: {msg_file}")
        print("Usage: python comprehensive_example.py <path_to_msg_file>")
        return

    print("🚀 Starting comprehensive email processing...")
    print(f"📧 Processing: {msg_file}")
    print()

    # Process the email and get comprehensive report
    report = process_email_comprehensive(msg_file)

    # Print the report
    print(report)

    # Optionally save to file
    output_file = f"email_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n💾 Report saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save report to file: {e}")


if __name__ == "__main__":
    main()
