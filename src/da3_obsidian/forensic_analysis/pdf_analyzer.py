"""
PDF-specific forensic analysis components.
"""

import io
import logging
import traceback
from typing import Any, Dict

import pdfplumber
import pikepdf
from pdf2image import convert_from_bytes

from .config import ForensicConfig

logger = logging.getLogger(__name__)


class PDFAnalyzer:
    """Handles PDF-specific forensic analysis."""

    def __init__(self, config: ForensicConfig):
        """
        Initialize PDF analyzer.

        Args:
            config: Forensic analysis configuration
        """
        self.config = config

    def analyze_pdf(self, file_path: str, file_bytes: bytes, results: Dict[str, Any]) -> None:
        """
        Comprehensive PDF analysis for fraud detection.

        Args:
            file_path: Path to PDF file
            file_bytes: PDF file content as bytes
            results: Results dictionary to update
        """
        logger.info(f"Starting PDF analysis: {file_path}")

        try:
            with pikepdf.Pdf.open(io.BytesIO(file_bytes)) as pdf:
                logger.debug("PDF opened with pikepdf")

                # Extract and analyze metadata
                self._analyze_pdf_metadata(pdf, results)

                # Analyze PDF structure
                self._analyze_pdf_structure(pdf, file_path, results)

                # Extract and analyze text
                self._analyze_pdf_text(file_path, results)

                # Extract and analyze embedded images
                self._analyze_pdf_images(pdf, results)

                # Analyze PDF pages as images
                self._analyze_pdf_page_images(pdf, results)

            logger.info("PDF analysis completed")

        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            self._add_analysis_error(results, "pdf_analysis", e)
            raise

    def _analyze_pdf_metadata(self, pdf: pikepdf.Pdf, results: Dict[str, Any]) -> None:
        """Analyze PDF metadata for inconsistencies."""
        try:
            metadata = pdf.docinfo

            if metadata:
                # Convert pikepdf.Dictionary to regular dict
                meta_dict = {str(k)[1:]: str(v) for k, v in metadata.items()}

                if "metadata_analysis" not in results:
                    results["metadata_analysis"] = {}

                results["metadata_analysis"]["pdf_metadata"] = meta_dict
                logger.info(f"Extracted PDF metadata: {len(meta_dict)} fields")

                # Check for metadata inconsistencies
                self._check_pdf_metadata_issues(meta_dict, results)
            else:
                if "metadata_analysis" not in results:
                    results["metadata_analysis"] = {}
                results["metadata_analysis"]["pdf_metadata"] = None

                issue = "Missing PDF metadata"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

        except Exception as e:
            logger.error(f"Error analyzing PDF metadata: {e}")
            self._add_analysis_error(results, "pdf_metadata", e)

    def _check_pdf_metadata_issues(self, metadata: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Check PDF metadata for inconsistencies and potential fraud indicators."""
        logger.debug("Analyzing PDF metadata")
        metadata_issues = []

        # Check creation date vs modification date
        create_date = metadata.get("CreationDate")
        mod_date = metadata.get("ModDate")

        if create_date and mod_date:
            # PDF dates are typically in format: D:20210818130512+00'00'
            # Simple string comparison for chronological order
            if create_date > mod_date:
                issue = "Creation date is later than modification date"
                logger.warning(f"Metadata issue detected: {issue}")
                metadata_issues.append(issue)

        # Check for missing essential metadata
        essential_fields = ["CreationDate", "Producer", "Creator"]
        missing_fields = [field for field in essential_fields if field not in metadata]

        if missing_fields:
            issue = f"Missing essential metadata: {', '.join(missing_fields)}"
            logger.warning(f"Metadata issue detected: {issue}")
            metadata_issues.append(issue)

        # Common fraud indicators in metadata
        suspicious_software = ["GIMP", "Photoshop", "Paint", "ABBYY", "Microsoft Paint"]
        for software in suspicious_software:
            if any(software.lower() in str(v).lower() for v in metadata.values()):
                issue = f"Document possibly edited with {software}, which can indicate manipulation"
                logger.warning(f"Metadata issue detected: {issue}")
                metadata_issues.append(issue)

        # Store the results
        if "metadata_analysis" not in results:
            results["metadata_analysis"] = {}

        results["metadata_analysis"]["metadata_issues"] = metadata_issues
        if metadata_issues:
            for issue in metadata_issues:
                self._add_fraud_indicator(results, issue)

        logger.info(f"Metadata analysis completed: found {len(metadata_issues)} issues")

    def _analyze_pdf_structure(self, pdf: pikepdf.Pdf, file_path: str, results: Dict[str, Any]) -> None:
        """Analyze PDF internal structure for anomalies."""
        try:
            if "integrity_analysis" not in results:
                results["integrity_analysis"] = {}

            results["integrity_analysis"]["pdf_structure"] = {}
            structure = results["integrity_analysis"]["pdf_structure"]

            # Check PDF version consistency
            self._check_version_consistency(pdf, file_path, structure, results)

            # Check for JavaScript (security risk)
            self._check_javascript_content(pdf, structure, results)

            # Check encryption status
            structure["is_encrypted"] = pdf.is_encrypted
            if pdf.is_encrypted:
                issue = "PDF is encrypted which may hinder complete analysis"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

            # Check for digital signatures
            self._check_digital_signatures(pdf, structure)

            # Check for incremental updates
            self._check_incremental_updates(file_path, structure, results)

            # Check for embedded files
            self._check_embedded_files(pdf, structure, results)

            # Analyze object structure
            self._analyze_pdf_objects(pdf, structure, results)

            # Check for optional content (layers)
            self._check_optional_content(pdf, structure, results)

            logger.info("PDF structure analysis completed")

        except Exception as e:
            logger.error(f"Error analyzing PDF structure: {e}")
            self._add_analysis_error(results, "pdf_structure", e)

    def _check_version_consistency(
        self, pdf: pikepdf.Pdf, file_path: str, structure: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Check PDF version consistency between header and catalog."""
        try:
            logger.debug("Checking PDF version consistency")
            header_version = None
            catalog_version = None

            # Extract PDF header version
            with open(file_path, "rb") as f:
                header_line = f.readline()
                if header_line.startswith(b"%PDF-"):
                    header_version = header_line[5:].strip().decode("utf-8", errors="ignore")

            # Extract catalog version
            if "/Version" in pdf.Root:
                catalog_version = str(pdf.Root["/Version"])[1:]  # Remove leading '/'

            if header_version and catalog_version and header_version != catalog_version:
                structure["version_mismatch"] = {"header_version": header_version, "catalog_version": catalog_version}

                issue = f"PDF version mismatch: header={header_version}, catalog={catalog_version}"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

        except Exception as e:
            logger.error(f"Error checking PDF version consistency: {e}")

    def _check_javascript_content(self, pdf: pikepdf.Pdf, structure: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Check for JavaScript content in PDF."""
        try:
            logger.debug("Checking for JavaScript in PDF")
            has_js = False
            js_content = []

            # Search for JavaScript in the entire PDF
            for obj_id, obj in enumerate(pdf.objects):
                if isinstance(obj, pikepdf.Object) and hasattr(obj, "read_bytes"):
                    try:
                        # Try to read content as bytes if possible
                        content = obj.read_bytes()
                        if isinstance(content, bytes) and (b"/JS" in content or b"/JavaScript" in content):
                            has_js = True
                            js_content.append(f"Object {obj_id}")
                    except Exception:
                        # Skip if can't read as bytes
                        pass
                elif isinstance(obj, dict):
                    # Check dictionary objects for JavaScript keys
                    if "/JS" in obj or "/JavaScript" in obj:
                        has_js = True
                        js_content.append(f"Object {obj_id}")

            structure["has_javascript"] = has_js
            if has_js:
                structure["javascript_locations"] = js_content

                issue = "PDF contains JavaScript which may indicate malicious content"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

        except Exception as e:
            logger.error(f"Error checking JavaScript content: {e}")

    def _check_digital_signatures(self, pdf: pikepdf.Pdf, structure: Dict[str, Any]) -> None:
        """Check for digital signatures."""
        try:
            logger.debug("Checking for digital signatures")
            has_signatures = False
            if "/AcroForm" in pdf.Root and "/Fields" in pdf.Root["/AcroForm"]:
                for field in pdf.Root["/AcroForm"]["/Fields"]:
                    if field.get("/FT") == "/Sig":
                        has_signatures = True
                        break

            structure["has_signatures"] = has_signatures
            logger.info(f"Digital signatures present: {has_signatures}")

        except Exception as e:
            logger.error(f"Error checking digital signatures: {e}")

    def _check_incremental_updates(self, file_path: str, structure: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Check for incremental updates indicating editing."""
        try:
            logger.debug("Checking for incremental updates")
            with open(file_path, "rb") as f:
                content = f.read()
                # Count occurrences of "%%EOF" which indicates multiple updates
                update_count = content.count(b"%%EOF")

                structure["incremental_updates"] = update_count - 1

                if update_count > 1:
                    issue = f"PDF has {update_count-1} incremental updates, indicating editing"
                    logger.warning(issue)
                    self._add_fraud_indicator(results, issue)

        except Exception as e:
            logger.error(f"Error checking incremental updates: {e}")
            # Non-critical analysis, continue

    def _check_embedded_files(self, pdf: pikepdf.Pdf, structure: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Check for embedded files."""
        try:
            logger.debug("Analyzing PDF objects")
            embedded_files = []
            if "/Names" in pdf.Root and "/EmbeddedFiles" in pdf.Root["/Names"]:
                names = pdf.Root["/Names"]["/EmbeddedFiles"]["/Names"]
                # Names is an array of alternating name, filespec pairs
                for i in range(0, len(names), 2):
                    if i + 1 < len(names):
                        filename = str(names[i])
                        embedded_files.append(filename)

            structure["embedded_files"] = embedded_files
            if embedded_files:
                issue = f"PDF contains {len(embedded_files)} embedded files which may indicate hidden content"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

        except Exception as e:
            logger.error(f"Error checking embedded files: {e}")

    def _analyze_pdf_objects(self, pdf: pikepdf.Pdf, structure: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Analyze PDF objects and filters."""
        try:
            logger.debug("Analyzing PDF filters")
            object_counts = {
                "total_objects": len(pdf.objects),
                "image_objects": 0,
                "form_objects": 0,
                "annotation_objects": 0,
                "embedded_files": 0,
            }

            # Check for unusual or suspicious filters
            filters_used = set()

            # Scan all objects for filters
            for obj_num, obj in enumerate(pdf.objects):
                try:
                    # Check if the object is dictionary-like (either a dict or a Stream)
                    if obj_num != 0 and isinstance(obj, (dict, pikepdf.Stream)):
                        if "/Filter" in obj:
                            filter_val = obj["/Filter"]
                            if isinstance(filter_val, list):
                                for f in filter_val:
                                    filters_used.add(str(f))
                            else:
                                filters_used.add(str(filter_val))
                except Exception:
                    # Skip objects that cause issues when accessing
                    pass

            structure["filters_used"] = list(filters_used)

            # Check if uncommon filters are used
            uncommon_filters = filters_used - {"/FlateDecode", "/DCTDecode"}  # Common filters
            if uncommon_filters:
                issue = f"PDF uses uncommon filters: {', '.join(uncommon_filters)}"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

            # Scan pages for objects
            logger.debug("Scanning PDF pages for objects")
            for page_idx, page in enumerate(pdf.pages):
                try:
                    if "/XObject" in page:
                        for _obj_name, obj in page["/XObject"].items():
                            if "/Subtype" in obj and obj["/Subtype"] == "/Image":
                                object_counts["image_objects"] += 1
                    if "/Annots" in page:
                        object_counts["annotation_objects"] += len(page["/Annots"])
                    if "/AcroForm" in page:
                        object_counts["form_objects"] += 1
                except Exception as e:
                    logger.error(f"Error scanning page {page_idx + 1} objects: {e}")
                    continue

            structure["object_counts"] = object_counts
            logger.info(
                f"Object counts: {object_counts['total_objects']} total, {object_counts['image_objects']} images"
            )

        except Exception as e:
            logger.error(f"Error analyzing PDF objects: {e}")

    def _check_optional_content(self, pdf: pikepdf.Pdf, structure: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Check for optional content (layers)."""
        try:
            logger.debug("Checking for optional content (layers)")
            has_optional_content = "/OCProperties" in pdf.Root
            structure["has_layers"] = has_optional_content

            if has_optional_content:
                # Count number of optional content groups
                ocg_count = 0
                if "/OCGs" in pdf.Root["/OCProperties"]:
                    ocg_count = len(pdf.Root["/OCProperties"]["/OCGs"])

                structure["layer_count"] = ocg_count

                issue = f"PDF contains {ocg_count} layers which may be used to hide content"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

        except Exception as e:
            logger.error(f"Error checking optional content: {e}")

    def _analyze_pdf_text(self, file_path: str, results: Dict[str, Any]) -> None:
        """Extract and analyze text content for inconsistencies."""
        try:
            logger.debug(f"Extracting and analyzing text from PDF: {file_path}")

            # Initialize text analysis section if not present
            if "text_analysis" not in results:
                results["text_analysis"] = {}

            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF opened with pdfplumber: {len(pdf.pages)} pages")
                text_blocks = []
                fonts_used = {}

                # Limit analysis to configured max pages
                max_pages = self.config.get("Analysis", "max_pdf_pages", 20)
                pages_to_analyze = min(len(pdf.pages), max_pages)

                if pages_to_analyze < len(pdf.pages):
                    logger.warning(f"Analyzing only {pages_to_analyze} of {len(pdf.pages)} pages (max_pdf_pages limit)")

                for page_num, page in enumerate(pdf.pages[:pages_to_analyze]):
                    logger.debug(f"Extracting text from page {page_num + 1}")
                    try:
                        text = page.extract_text()
                        if text:
                            text_blocks.append(
                                {
                                    "page": page_num + 1,
                                    "text": text[:500],  # Limit text stored in results
                                    "char_count": len(text),
                                }
                            )

                        # Extract font information
                        if hasattr(page, "chars") and page.chars:
                            for char in page.chars:
                                font_name = char.get("fontname", "unknown")
                                if font_name in fonts_used:
                                    fonts_used[font_name] += 1
                                else:
                                    fonts_used[font_name] = 1
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                        continue

                results["text_analysis"]["text_extraction"] = {
                    "pages_with_text": len(text_blocks),
                    "total_text_length": sum(block["char_count"] for block in text_blocks),
                    "text_blocks": text_blocks[:2],  # Just include first couple of blocks as examples
                }

                results["text_analysis"]["font_analysis"] = {
                    "distinct_fonts": len(fonts_used),
                    "font_distribution": fonts_used,
                }

                logger.info(
                    f"Text extraction completed: {len(text_blocks)} pages with text, {len(fonts_used)} distinct fonts"
                )

                # Analyze font consistency
                font_threshold = self.config.get("Thresholds", "font_count_threshold", 3)
                if len(fonts_used) > font_threshold:
                    issue = f"Document uses {len(fonts_used)} different fonts, which may indicate cut-and-paste forgery"
                    logger.warning(f"Font issue detected: {issue}")
                    self._add_fraud_indicator(results, issue)

        except Exception as e:
            error_details = {"message": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
            logger.error(f"Error in PDF text analysis: {error_details['error_type']}: {error_details['message']}")
            results["text_analysis"]["analysis_error"] = error_details

    def _analyze_pdf_images(self, pdf: pikepdf.Pdf, results: Dict[str, Any]) -> None:
        """Extract and analyze embedded images."""
        try:
            logger.debug("Extracting and analyzing images from PDF")

            # Initialize image analysis section if not present
            if "image_analysis" not in results:
                results["image_analysis"] = {}

            image_analysis = {"total_images": 0, "images_analyzed": 0, "suspicious_images": []}

            # Limit analysis to configured max images
            max_images = self.config.get("Analysis", "max_images", 50)

            for page_idx, page in enumerate(pdf.pages):
                logger.debug(f"Examining page {page_idx + 1} for images")

                if "/XObject" not in page:
                    continue

                for obj_name, obj in page["/XObject"].items():
                    if "/Subtype" not in obj or obj["/Subtype"] != "/Image":
                        continue

                    image_analysis["total_images"] += 1

                    # Check if we've reached the maximum number of images to analyze
                    if image_analysis["images_analyzed"] >= max_images:
                        logger.warning(f"Reached maximum image analysis limit ({max_images})")
                        break

                    # Analyze basic image properties for first few images
                    try:
                        # Basic image properties
                        width = obj.get("/Width", 0)
                        height = obj.get("/Height", 0)
                        color_space = str(obj.get("/ColorSpace", "Unknown"))
                        bits_per_component = obj.get("/BitsPerComponent", 0)

                        image_info = {
                            "page": page_idx + 1,
                            "name": str(obj_name),
                            "width": width,
                            "height": height,
                            "color_space": color_space,
                            "bits_per_component": bits_per_component,
                        }

                        logger.debug(f"Found image on page {page_idx + 1}: {width}x{height}")

                        # Look for suspicious properties
                        if width < 100 or height < 100:
                            image_info["suspicious_reason"] = "Very small image (possible hidden content)"
                            image_analysis["suspicious_images"].append(image_info)
                            logger.warning(
                                f"Suspicious image found on page {page_idx + 1}: {image_info['suspicious_reason']}"
                            )

                        image_analysis["images_analyzed"] += 1

                    except Exception as e:
                        logger.error(f"Error analyzing image on page {page_idx + 1}: {e}")
                        # Continue to next image on error
                        pass

            results["image_analysis"]["pdf_images"] = image_analysis
            logger.info(
                f"PDF image analysis complete: {image_analysis['total_images']} total images, {len(image_analysis['suspicious_images'])} suspicious"
            )

            if image_analysis["suspicious_images"]:
                issue = f"Found {len(image_analysis['suspicious_images'])} suspicious images in PDF"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

        except Exception as e:
            error_details = {"message": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
            logger.error(f"Error in PDF image extraction: {error_details['error_type']}: {error_details['message']}")
            results["image_analysis"]["analysis_error"] = error_details

    def _analyze_pdf_page_images(self, pdf: pikepdf.Pdf, results: Dict[str, Any]) -> None:
        """Convert PDF pages to images and analyze for manipulation."""
        try:
            logger.info("Extracting PDF pages as images for analysis")

            # Initialize page image analysis results
            if "image_analysis" not in results:
                results["image_analysis"] = {}

            results["image_analysis"]["page_images"] = {"pages_analyzed": 0, "pages_with_issues": 0, "page_results": []}

            # Track issues across all pages
            suspicious_pages = []
            all_issues = []

            # Get maximum pages to analyze from configuration
            max_pages = self.config.get("Analysis", "max_pdf_pages", 20)

            # Save pikepdf object to a BytesIO buffer to use with pdf2image
            pdf_buffer = io.BytesIO()
            pdf.save(pdf_buffer)
            pdf_buffer.seek(0)

            # Convert PDF to a list of PIL images
            try:
                logger.debug("Converting PDF pages to images")
                images = convert_from_bytes(pdf_buffer.getvalue(), dpi=200)

                # Limit to maximum pages
                if len(images) > max_pages:
                    logger.warning(f"Limiting PDF page image analysis to {max_pages} pages (of {len(images)})")
                    images = images[:max_pages]

                logger.info(f"Successfully converted {len(images)} PDF pages to images")
            except Exception as e:
                logger.error(f"Error converting PDF to images: {e}")
                results["image_analysis"]["page_images"]["error"] = {"message": str(e), "error_type": type(e).__name__}
                return

            # Import ImageAnalyzer here to avoid circular imports
            from .image_analyzer import ImageAnalyzer

            image_analyzer = ImageAnalyzer(self.config)

            for page_idx, img in enumerate(images):
                logger.info(f"Analyzing PDF page {page_idx + 1} as image")

                # Create a buffer to save the image temporarily
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG", quality=95)
                img_buffer.seek(0)

                # Create a temporary result dictionary for this page
                page_results = {
                    "file_info": {},
                    "metadata_analysis": {},
                    "integrity_analysis": {},
                    "text_analysis": {},
                    "image_analysis": {},
                    "summary": {"risk_score": 0, "potential_fraud_indicators": []},
                }

                # Run image analysis on this page
                try:
                    image_analyzer.analyze_image(
                        file_path=f"page_{page_idx+1}.jpg",  # Virtual filename
                        file_bytes=img_buffer.getvalue(),
                        results=page_results,
                        is_virtual_file=True,
                        from_pdf_page=True,
                    )

                    # Extract important analysis results
                    fraud_indicators = page_results["summary"]["potential_fraud_indicators"]

                    # Add prefix to all indicators to identify the source page
                    prefixed_indicators = [f"Page {page_idx+1}: {indicator}" for indicator in fraud_indicators]

                    # Store page results with a summary
                    page_summary = {
                        "page_number": page_idx + 1,
                        "risk_score": page_results["summary"]["risk_score"],
                        "fraud_indicators": fraud_indicators,
                        "analysis_results": {
                            # Add key analysis results that indicate manipulation
                            "copy_move": page_results["image_analysis"].get("copy_move_detection", {}),
                            "noise": page_results["image_analysis"].get("noise_analysis", {}),
                            "edge_integrity": page_results["image_analysis"].get("edge_integrity", {}),
                        },
                    }

                    results["image_analysis"]["page_images"]["page_results"].append(page_summary)
                    results["image_analysis"]["page_images"]["pages_analyzed"] += 1

                    # If the page has fraud indicators, add it to the list of suspicious pages
                    if fraud_indicators:
                        suspicious_pages.append(page_idx + 1)
                        all_issues.extend(prefixed_indicators)
                except Exception as e:
                    logger.error(f"Error analyzing page {page_idx + 1}: {e}")
                    results["image_analysis"]["page_images"]["page_results"].append(
                        {"page_number": page_idx + 1, "error": str(e), "error_type": type(e).__name__}
                    )

            # Update summary with page image analysis findings
            if suspicious_pages:
                results["image_analysis"]["page_images"]["pages_with_issues"] = len(suspicious_pages)

                issue = f"Image analysis found potential manipulation on pages: {', '.join(map(str, suspicious_pages))}"
                logger.warning(issue)
                self._add_fraud_indicator(results, issue)

                # Add specific issues found (limited to avoid overwhelming the summary)
                if len(all_issues) > 5:
                    # Just add a summary if there are many issues
                    issue = f"Found {len(all_issues)} image manipulation indicators across pages (see detailed results)"
                    logger.warning(issue)
                    self._add_fraud_indicator(results, issue)
                else:
                    # Add all issues if there are just a few
                    logger.warning(f"Adding {len(all_issues)} specific page issues to summary")
                    for issue in all_issues:
                        self._add_fraud_indicator(results, issue)

            logger.info(f"PDF page image analysis complete: {len(suspicious_pages)} pages with issues")

        except Exception as e:
            logger.error(f"Error analyzing PDF pages as images: {e}")
            error_details = {"message": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
            if "image_analysis" not in results:
                results["image_analysis"] = {}
            results["image_analysis"]["page_images_error"] = error_details

    def _add_fraud_indicator(self, results: Dict[str, Any], indicator: str) -> None:
        """Add a fraud indicator to the results."""
        if "summary" not in results:
            results["summary"] = {"potential_fraud_indicators": []}
        results["summary"]["potential_fraud_indicators"].append(indicator)
        logger.warning(f"Fraud indicator: {indicator}")

    def _add_analysis_error(self, results: Dict[str, Any], analysis_type: str, error: Exception) -> None:
        """Add analysis error to results."""
        error_details = {"message": str(error), "error_type": type(error).__name__, "traceback": traceback.format_exc()}

        if analysis_type not in results:
            results[analysis_type] = {}
        results[analysis_type]["analysis_error"] = error_details
