"""
Metadata analysis components for forensic analysis.
"""

import logging
from typing import Any, Dict, List

from .config import ForensicConfig

logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """Handles metadata analysis for fraud detection."""

    def __init__(self, config: ForensicConfig):
        """
        Initialize metadata analyzer.

        Args:
            config: Forensic analysis configuration
        """
        self.config = config

        # Define suspicious software patterns
        self.suspicious_software = [
            "GIMP",
            "Photoshop",
            "Lightroom",
            "Paint",
            "ABBYY",
            "Inkscape",
            "CorelDRAW",
            "Pixlr",
            "Capture One",
            "Darktable",
            "Paint.NET",
            "PhotoScape",
            "Pixelmator",
            "Snapseed",
            "Microsoft Paint",
        ]

    def analyze_pdf_metadata(self, metadata: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Analyze PDF metadata for inconsistencies and fraud indicators.

        Args:
            metadata: PDF metadata dictionary
            results: Results dictionary to store findings
        """
        logger.debug("Analyzing PDF metadata")
        metadata_issues = []

        # Check creation date vs modification date
        create_date = metadata.get("CreationDate")
        mod_date = metadata.get("ModDate")

        if create_date and mod_date:
            # Simple string comparison for chronological order
            if create_date > mod_date:
                issue = "Creation date is later than modification date"
                logger.warning(f"Metadata issue: {issue}")
                metadata_issues.append(issue)

        # Check for missing essential metadata
        essential_fields = ["CreationDate", "Producer", "Creator"]
        missing_fields = [field for field in essential_fields if field not in metadata]

        if missing_fields:
            issue = f"Missing essential metadata: {', '.join(missing_fields)}"
            logger.warning(f"Metadata issue: {issue}")
            metadata_issues.append(issue)

        # Check for suspicious software
        for software in self.suspicious_software:
            if any(software.lower() in str(v).lower() for v in metadata.values()):
                issue = f"Document possibly edited with {software}, which can indicate manipulation"
                logger.warning(f"Metadata issue: {issue}")
                metadata_issues.append(issue)

        # Store results
        self._store_metadata_issues(results, metadata_issues)

        logger.info(f"PDF metadata analysis: {len(metadata_issues)} issues found")

    def analyze_image_metadata(self, metadata: Dict[str, str], results: Dict[str, Any]) -> None:
        """
        Analyze image EXIF metadata for inconsistencies.

        Args:
            metadata: Image EXIF metadata dictionary
            results: Results dictionary to store findings
        """
        logger.debug("Analyzing image metadata")
        metadata_issues = []

        # Check for common metadata fields
        essential_fields = ["EXIF DateTimeOriginal", "Image Make", "Image Model"]
        missing_fields = [field for field in essential_fields if field not in metadata]

        if missing_fields:
            issue = f"Missing common metadata: {', '.join(missing_fields)}"
            logger.warning(f"Metadata issue: {issue}")
            metadata_issues.append(issue)

        # Check for date consistencies
        date_fields = ["EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"]

        date_values = [metadata.get(field) for field in date_fields if field in metadata]
        if len(set(date_values)) > 1 and len(date_values) > 1:
            issue = "Inconsistent dates in metadata fields"
            logger.warning(f"Metadata issue: {issue}")
            metadata_issues.append(issue)

        # Check for editing software
        software_fields = ["Image Software", "EXIF Software"]
        editing_software = [metadata.get(field) for field in software_fields if field in metadata]

        for software in self.suspicious_software:
            if any(software in str(sw) for sw in editing_software if sw):
                issue = f"Image edited with {software}, which can indicate manipulation"
                logger.warning(f"Metadata issue: {issue}")
                metadata_issues.append(issue)

        # Check for GPS data inconsistencies
        self._check_gps_consistency(metadata, metadata_issues)

        # Check camera-specific inconsistencies
        self._check_camera_consistency(metadata, metadata_issues)

        # Store results
        self._store_metadata_issues(results, metadata_issues)

        logger.info(f"Image metadata analysis: {len(metadata_issues)} issues found")

    def _check_gps_consistency(self, metadata: Dict[str, str], issues: List[str]) -> None:
        """Check GPS metadata for consistency."""
        try:
            gps_fields = [
                "GPS GPSLatitude",
                "GPS GPSLongitude",
                "GPS GPSLatitudeRef",
                "GPS GPSLongitudeRef",
                "GPS GPSTimeStamp",
                "GPS GPSDateStamp",
            ]

            present_gps_fields = [field for field in gps_fields if field in metadata]

            # If some GPS fields are present but not others, it might indicate tampering
            if 0 < len(present_gps_fields) < len(gps_fields):
                missing_gps = [field for field in gps_fields if field not in metadata]
                if len(missing_gps) > 2:  # Allow for some optional fields
                    issue = f"Incomplete GPS data: missing {', '.join(missing_gps[:3])}"
                    logger.warning(f"GPS metadata issue: {issue}")
                    issues.append(issue)

            # Check for impossible GPS coordinates
            lat_field = metadata.get("GPS GPSLatitude")
            lon_field = metadata.get("GPS GPSLongitude")

            if lat_field and lon_field:
                try:
                    # Simple check for obviously invalid coordinates
                    if "inf" in str(lat_field).lower() or "inf" in str(lon_field).lower():
                        issue = "Invalid GPS coordinates detected"
                        logger.warning(f"GPS metadata issue: {issue}")
                        issues.append(issue)
                except Exception:
                    # If we can't parse the coordinates, skip this check
                    pass

        except Exception as e:
            logger.error(f"Error checking GPS consistency: {e}")

    def _check_camera_consistency(self, metadata: Dict[str, str], issues: List[str]) -> None:
        """Check camera-specific metadata for consistency."""
        try:
            # Check for make/model consistency
            make = metadata.get("Image Make", "").strip()
            model = metadata.get("Image Model", "").strip()

            if make and model:
                # Check if model name contains make name (common pattern)
                if make.lower() not in model.lower() and len(make) > 3:
                    # Some exceptions for common manufacturer variations
                    make_variations = {
                        "canon": ["canon"],
                        "nikon": ["nikon"],
                        "sony": ["sony", "ilce", "dsc"],
                        "apple": ["iphone", "ipad"],
                        "samsung": ["sm-", "galaxy"],
                        "google": ["pixel"],
                        "oneplus": ["oneplus"],
                    }

                    make_lower = make.lower()
                    model_lower = model.lower()

                    # Check if any variation matches
                    variation_found = False
                    for manufacturer, variations in make_variations.items():
                        if manufacturer in make_lower:
                            if any(var in model_lower for var in variations):
                                variation_found = True
                                break

                    if not variation_found:
                        issue = f"Camera make '{make}' and model '{model}' may be inconsistent"
                        logger.warning(f"Camera metadata issue: {issue}")
                        issues.append(issue)

            # Check for unusual EXIF values
            iso = metadata.get("EXIF ISOSpeedRatings")
            if iso:
                try:
                    iso_value = int(str(iso))
                    if iso_value > 102400 or iso_value < 50:  # Extreme ISO values
                        issue = f"Unusual ISO value: {iso_value}"
                        logger.warning(f"Camera metadata issue: {issue}")
                        issues.append(issue)
                except (ValueError, TypeError):
                    pass

            # Check flash consistency
            flash = metadata.get("EXIF Flash")
            flash_mode = metadata.get("EXIF FlashMode")

            if flash and flash_mode:
                # Simple consistency check - if flash fired, mode should be consistent
                flash_str = str(flash).lower()
                mode_str = str(flash_mode).lower()

                if "no flash" in flash_str and "auto" in mode_str:
                    # This could be normal, but log for review
                    logger.debug(f"Flash inconsistency check: flash='{flash}', mode='{flash_mode}'")

        except Exception as e:
            logger.error(f"Error checking camera consistency: {e}")

    def _store_metadata_issues(self, results: Dict[str, Any], issues: List[str]) -> None:
        """Store metadata issues in results and update fraud indicators."""
        if "metadata_analysis" not in results:
            results["metadata_analysis"] = {}

        results["metadata_analysis"]["metadata_issues"] = issues

        if issues:
            if "summary" not in results:
                results["summary"] = {"potential_fraud_indicators": []}

            # Add issues to fraud indicators
            results["summary"]["potential_fraud_indicators"].extend(issues)

    def get_metadata_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of metadata fields.

        Args:
            metadata: Metadata dictionary

        Returns:
            Summary dictionary with key statistics
        """
        summary = {
            "total_fields": len(metadata),
            "has_creation_date": False,
            "has_software_info": False,
            "has_camera_info": False,
            "has_gps_info": False,
        }

        # Check for key field categories
        creation_fields = ["CreationDate", "EXIF DateTimeOriginal", "Image DateTime"]
        software_fields = ["Producer", "Creator", "Image Software", "EXIF Software"]
        camera_fields = ["Image Make", "Image Model", "EXIF Model"]
        gps_fields = ["GPS GPSLatitude", "GPS GPSLongitude"]

        summary["has_creation_date"] = any(field in metadata for field in creation_fields)
        summary["has_software_info"] = any(field in metadata for field in software_fields)
        summary["has_camera_info"] = any(field in metadata for field in camera_fields)
        summary["has_gps_info"] = any(field in metadata for field in gps_fields)

        return summary

    def extract_key_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metadata fields for analysis.

        Args:
            metadata: Full metadata dictionary

        Returns:
            Dictionary with key metadata fields
        """
        key_fields = {
            # PDF fields
            "CreationDate": metadata.get("CreationDate"),
            "ModDate": metadata.get("ModDate"),
            "Producer": metadata.get("Producer"),
            "Creator": metadata.get("Creator"),
            # Image fields
            "DateTime": metadata.get("EXIF DateTimeOriginal") or metadata.get("Image DateTime"),
            "Make": metadata.get("Image Make"),
            "Model": metadata.get("Image Model"),
            "Software": metadata.get("Image Software") or metadata.get("EXIF Software"),
            "GPS": {"Latitude": metadata.get("GPS GPSLatitude"), "Longitude": metadata.get("GPS GPSLongitude")},
        }

        # Remove None values
        return {k: v for k, v in key_fields.items() if v is not None}
