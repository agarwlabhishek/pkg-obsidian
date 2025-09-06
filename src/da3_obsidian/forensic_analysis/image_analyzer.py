"""
Image-specific forensic analysis components.
"""

import io
import logging
import traceback
from typing import Any, Dict

import cv2
import exifread
import numpy as np
from PIL import Image, ImageChops

from .config import ForensicConfig

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Handles image-specific forensic analysis."""

    def __init__(self, config: ForensicConfig):
        """
        Initialize image analyzer.

        Args:
            config: Forensic analysis configuration
        """
        self.config = config

    def analyze_image(
        self,
        file_path: str,
        file_bytes: bytes,
        results: Dict[str, Any],
        is_virtual_file: bool = False,
        from_pdf_page: bool = False,
    ) -> None:
        """
        Comprehensive image analysis for fraud detection.

        Args:
            file_path: Path to image file (or virtual name)
            file_bytes: Image file content as bytes
            results: Results dictionary to update
            is_virtual_file: Whether file exists only in memory
            from_pdf_page: Whether image is from PDF page extraction
        """
        logger.info(f"Starting image analysis: {file_path} (virtual={is_virtual_file}, from_pdf={from_pdf_page})")

        try:
            if "image_analysis" not in results:
                results["image_analysis"] = {}

            # Extract EXIF metadata (skip for PDF-extracted images)
            if not from_pdf_page:
                self._extract_and_analyze_exif(file_bytes, results)

            # Open image for pixel-level analysis
            with Image.open(io.BytesIO(file_bytes)) as img:
                # Store basic properties
                results["image_analysis"]["basic_properties"] = {
                    "format": img.format,
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height,
                    "dpi": img.info.get("dpi"),
                }

                logger.info(f"Image properties: {img.format}, {img.width}x{img.height}, {img.mode}")

                # Convert to OpenCV format
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_cv = img_array

                # Core forensic analyses
                self._perform_copy_move_detection(img_cv, results)
                self._perform_noise_analysis(img_cv, results)
                self._perform_edge_integrity_analysis(img_cv, results)

                # JPEG-specific analyses (skip for PDF-extracted images)
                if img.format == "JPEG" and not from_pdf_page:
                    self._perform_ela_analysis(img, results)
                    self._perform_jpeg_ghost_analysis(img_cv, results)

                    if not is_virtual_file:
                        self._analyze_jpeg_quantization_from_file(file_path, results)
                    else:
                        self._analyze_jpeg_quantization_from_bytes(file_bytes, results)

                # Color image analyses
                if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                    self._perform_shadow_analysis(img_cv, results)

                    # CFA analysis (skip for PDF-extracted images)
                    if not from_pdf_page:
                        self._perform_cfa_analysis(img_cv, results)

            logger.info("Image analysis completed")

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            self._add_analysis_error(results, "image_analysis", e)
            raise

    def _extract_and_analyze_exif(self, file_bytes: bytes, results: Dict[str, Any]) -> None:
        """Extract and analyze EXIF metadata."""
        try:
            logger.debug("Extracting and analyzing EXIF metadata")
            exif_tags = exifread.process_file(io.BytesIO(file_bytes))

            if exif_tags:
                exif_dict = {str(k): str(v) for k, v in exif_tags.items()}

                if "metadata_analysis" not in results:
                    results["metadata_analysis"] = {}

                results["metadata_analysis"]["image_metadata"] = exif_dict
                self._analyze_image_metadata(exif_dict, results)
            else:
                if "metadata_analysis" not in results:
                    results["metadata_analysis"] = {}
                results["metadata_analysis"]["image_metadata"] = None

                self._add_fraud_indicator(results, "Image has no EXIF metadata, which may indicate it was stripped")

        except Exception as e:
            logger.error(f"Error extracting EXIF metadata: {e}")
            if "metadata_analysis" not in results:
                results["metadata_analysis"] = {}
            results["metadata_analysis"]["metadata_error"] = str(e)

    def _analyze_image_metadata(self, metadata: Dict[str, str], results: Dict[str, Any]) -> None:
        """Analyze image EXIF metadata for inconsistencies."""
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

        suspicious_software = [
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
        ]

        for software in suspicious_software:
            if any(software in str(sw) for sw in editing_software if sw):
                issue = f"Image edited with {software}, which can indicate manipulation"
                logger.warning(f"Metadata issue: {issue}")
                metadata_issues.append(issue)

        # Store the results
        if "metadata_analysis" not in results:
            results["metadata_analysis"] = {}

        results["metadata_analysis"]["metadata_issues"] = metadata_issues
        if metadata_issues:
            for issue in metadata_issues:
                self._add_fraud_indicator(results, issue)

        logger.info(f"Image metadata analysis: {len(metadata_issues)} issues found")

    def _perform_ela_analysis(self, img: Image.Image, results: Dict[str, Any]) -> None:
        """Perform Error Level Analysis to detect manipulation."""
        try:
            logger.debug("Performing Error Level Analysis (ELA)")

            temp_buffer = io.BytesIO()

            # Get configured JPEG quality
            jpeg_quality = self.config.get("Analysis", "jpeg_quality", 95)

            # Save with specific quality level
            img.save(temp_buffer, format="JPEG", quality=jpeg_quality)
            temp_buffer.seek(0)
            saved_img = Image.open(temp_buffer)

            # Calculate difference
            ela_img = ImageChops.difference(img, saved_img)

            # Amplify difference for analysis
            extrema = ela_img.getextrema()
            max_diff = max([ex[1] for ex in extrema])

            # Convert to numpy for statistics
            ela_array = np.array(ela_img)

            if len(ela_array.shape) == 3:  # Color image
                channel_means = [np.mean(ela_array[:, :, i]) for i in range(ela_array.shape[2])]
                overall_mean = np.mean(channel_means)
                channel_stds = [np.std(ela_array[:, :, i]) for i in range(ela_array.shape[2])]
                overall_std = np.mean(channel_stds)
            else:  # Grayscale
                overall_mean = np.mean(ela_array)
                overall_std = np.std(ela_array)

            # Get thresholds from configuration
            ela_mean_threshold = self.config.get("Thresholds", "ela_mean_threshold", 5.0)
            ela_std_threshold = self.config.get("Thresholds", "ela_std_threshold", 10.0)

            # Detect unusual variation
            suspicious = overall_mean > ela_mean_threshold or overall_std > ela_std_threshold or max_diff > 50

            result = {
                "performed": True,
                "max_difference": int(max_diff),
                "mean_difference": float(overall_mean),
                "std_deviation": float(overall_std),
                "suspicious": suspicious,
            }

            results["image_analysis"]["error_level_analysis"] = result

            if suspicious:
                self._add_fraud_indicator(results, "Error Level Analysis indicates possible image manipulation")

            logger.info(f"ELA: max_diff={max_diff}, mean={overall_mean:.2f}, suspicious={suspicious}")

        except Exception as e:
            logger.error(f"Error in ELA analysis: {e}")
            results["image_analysis"]["error_level_analysis"] = {
                "performed": False,
                "error": str(e),
                "suspicious": False,
            }

    def _perform_copy_move_detection(self, img_cv: np.ndarray, results: Dict[str, Any]) -> None:
        """Detect copy-move forgery using feature matching."""
        try:
            logger.debug("Detecting copy-move forgery")

            # Convert to grayscale
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv

            # Try different feature detectors
            detector = None
            detector_used = ""

            try:
                detector = cv2.SIFT_create()
                detector_used = "SIFT"
            except Exception:
                logger.warning("SIFT not available, trying ORB")
                try:
                    detector = cv2.ORB_create()
                    detector_used = "ORB"
                except Exception:
                    results["image_analysis"]["copy_move_detection"] = {
                        "performed": False,
                        "reason": "Required OpenCV feature detection algorithms not available",
                        "suspicious": False,
                    }
                    return

            logger.debug(f"Using {detector_used} for feature detection")
            keypoints, descriptors = detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 10:
                logger.info(f"Copy-move detection: not enough keypoints ({len(keypoints) if keypoints else 0})")
                results["image_analysis"]["copy_move_detection"] = {
                    "performed": True,
                    "keypoints_found": len(keypoints) if keypoints else 0,
                    "suspicious": False,
                    "reason": "Not enough keypoints for analysis",
                }
                return

            # Match descriptors
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(descriptors, descriptors, k=2)

            # Filter good matches
            good_matches = []
            for match in matches:
                # Skip self-matches
                if match[0].distance < 0.7 * match[1].distance and match[0].queryIdx != match[0].trainIdx:
                    # Calculate distance between matched keypoints
                    idx1 = match[0].queryIdx
                    idx2 = match[0].trainIdx
                    pt1 = keypoints[idx1].pt
                    pt2 = keypoints[idx2].pt

                    # Filter out very close points
                    distance = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
                    if distance > 20:  # Arbitrary threshold
                        good_matches.append(match)

            # Get threshold from configuration
            copy_move_threshold = self.config.get("Thresholds", "copy_move_threshold", 10)

            match_count = len(good_matches)
            suspicious = match_count > copy_move_threshold

            result = {
                "performed": True,
                "keypoints_found": len(keypoints),
                "similar_regions_found": match_count,
                "suspicious": suspicious,
                "reason": (
                    "Multiple similar non-adjacent regions detected"
                    if suspicious
                    else "No suspicious patterns detected"
                ),
            }

            results["image_analysis"]["copy_move_detection"] = result

            if suspicious:
                self._add_fraud_indicator(results, "Possible copy-move forgery detected in image")

            logger.info(f"Copy-move: {match_count} matches, suspicious={suspicious}")

        except Exception as e:
            logger.error(f"Error in copy-move detection: {e}")
            results["image_analysis"]["copy_move_detection"] = {
                "performed": False,
                "error": str(e),
                "suspicious": False,
            }

    def _perform_noise_analysis(self, img_cv: np.ndarray, results: Dict[str, Any]) -> None:
        """Analyze noise patterns for inconsistencies."""
        try:
            logger.debug("Analyzing image noise patterns")

            # Convert to grayscale
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv

            # Apply noise extraction filter
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.subtract(gray, blurred)

            # Divide into regions (4x4 grid)
            h, w = noise.shape
            regions = []
            rows, cols = 4, 4
            region_h, region_w = h // rows, w // cols

            for i in range(rows):
                for j in range(cols):
                    y1, y2 = i * region_h, (i + 1) * region_h
                    x1, x2 = j * region_w, (j + 1) * region_w

                    if y2 > h:
                        y2 = h
                    if x2 > w:
                        x2 = w

                    region = noise[y1:y2, x1:x2]
                    mean = np.mean(region)
                    std = np.std(region)

                    regions.append({"position": (i, j), "mean_noise": float(mean), "std_noise": float(std)})

            # Analyze variation across regions
            noise_means = [r["mean_noise"] for r in regions]
            noise_stds = [r["std_noise"] for r in regions]

            overall_std_of_means = np.std(noise_means)
            overall_std_of_stds = np.std(noise_stds)

            # Get thresholds from configuration
            noise_mean_threshold = self.config.get("Thresholds", "noise_mean_threshold", 2.0)
            noise_std_threshold = self.config.get("Thresholds", "noise_std_threshold", 3.0)

            # Determine if noise is inconsistent
            inconsistent_noise = (
                overall_std_of_means > noise_mean_threshold or overall_std_of_stds > noise_std_threshold
            )

            result = {
                "performed": True,
                "region_count": len(regions),
                "noise_mean_variation": float(overall_std_of_means),
                "noise_std_variation": float(overall_std_of_stds),
                "inconsistent_noise": inconsistent_noise,
            }

            results["image_analysis"]["noise_analysis"] = result

            if inconsistent_noise:
                self._add_fraud_indicator(
                    results, "Inconsistent noise patterns detected in image, possible manipulation"
                )

            logger.info(f"Noise: mean_var={overall_std_of_means:.2f}, inconsistent={inconsistent_noise}")

        except Exception as e:
            logger.error(f"Error in noise analysis: {e}")
            results["image_analysis"]["noise_analysis"] = {"performed": False, "error": str(e), "suspicious": False}

    def _perform_edge_integrity_analysis(self, img_cv: np.ndarray, results: Dict[str, Any]) -> None:
        """Analyze edge integrity to detect splicing."""
        try:
            logger.debug("Analyzing edge integrity")

            # Convert to grayscale
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv

            # Edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

            # Count suspicious lines
            suspicious_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    # Long straight lines can indicate splicing
                    if length > min(gray.shape) / 4:
                        suspicious_lines += 1

            # Check for pixel value discontinuities at edges
            edge_discontinuities = 0
            dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

            for direction in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                dx, dy = direction
                shifted = np.roll(gray, shift=(dy, dx), axis=(0, 1))
                diff = np.abs(gray.astype(np.float32) - shifted.astype(np.float32))

                large_diffs = np.logical_and(dilated_edges > 0, diff > 30)
                edge_discontinuities += np.sum(large_diffs)

            # Normalize by edge length
            edge_length = np.sum(edges > 0)
            discontinuity_ratio = edge_discontinuities / (edge_length * 4) if edge_length > 0 else 0

            # Get thresholds from configuration
            straight_lines_threshold = self.config.get("Thresholds", "straight_lines_threshold", 5)
            discontinuity_ratio_threshold = self.config.get("Thresholds", "discontinuity_ratio_threshold", 0.2)

            suspicious = (
                suspicious_lines > straight_lines_threshold or discontinuity_ratio > discontinuity_ratio_threshold
            )

            result = {
                "performed": True,
                "straight_lines": int(suspicious_lines) if lines is not None else 0,
                "discontinuity_ratio": float(discontinuity_ratio),
                "suspicious": suspicious,
                "reason": "Potential splicing artifacts detected" if suspicious else "No obvious splicing artifacts",
            }

            results["image_analysis"]["edge_integrity"] = result

            if suspicious:
                self._add_fraud_indicator(results, "Edge analysis detected potential splicing artifacts")

            logger.info(
                f"Edge integrity: lines={suspicious_lines}, disc_ratio={discontinuity_ratio:.3f}, suspicious={suspicious}"
            )

        except Exception as e:
            logger.error(f"Error in edge integrity analysis: {e}")
            results["image_analysis"]["edge_integrity"] = {"performed": False, "error": str(e), "suspicious": False}

    def _perform_jpeg_ghost_analysis(self, img_cv: np.ndarray, results: Dict[str, Any]) -> None:
        """Detect JPEG ghosts indicating manipulation."""
        try:
            logger.debug("Detecting JPEG ghosts")

            # Convert to grayscale
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv

            # Simplified JPEG ghost detection using DCT analysis
            block_size = 8  # JPEG uses 8x8 blocks
            h, w = gray.shape
            ghost_blocks = 0
            total_blocks = 0

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    total_blocks += 1
                    block = gray[y : y + block_size, x : x + block_size]

                    # DCT transform (similar to JPEG compression)
                    dct = cv2.dct(np.float32(block))

                    # Check for unusual coefficient patterns
                    if np.sum(np.abs(dct[:4, :4])) < 0.1 * np.sum(np.abs(dct)):
                        ghost_blocks += 1

            ghost_ratio = ghost_blocks / max(total_blocks, 1)

            # Get threshold from configuration
            ghost_ratio_threshold = self.config.get("Thresholds", "ghost_ratio_threshold", 0.05)

            ghosts_detected = ghost_ratio > ghost_ratio_threshold

            result = {
                "performed": True,
                "ghost_blocks": ghost_blocks,
                "total_blocks": total_blocks,
                "ghost_ratio": float(ghost_ratio),
                "ghosts_detected": ghosts_detected,
            }

            results["image_analysis"]["jpeg_ghost_analysis"] = result

            if ghosts_detected:
                self._add_fraud_indicator(
                    results, "JPEG ghost analysis indicates parts of the image may have been resaved"
                )

            logger.info(f"JPEG ghosts: ratio={ghost_ratio:.3f}, detected={ghosts_detected}")

        except Exception as e:
            logger.error(f"Error in JPEG ghost detection: {e}")
            results["image_analysis"]["jpeg_ghost_analysis"] = {
                "performed": False,
                "error": str(e),
                "suspicious": False,
            }

    def _analyze_jpeg_quantization_from_file(self, file_path: str, results: Dict[str, Any]) -> None:
        """Analyze JPEG quantization tables from file."""
        try:
            logger.debug(f"Analyzing JPEG quantization tables from file: {file_path}")

            with open(file_path, "rb") as f:
                file_bytes = f.read()

            self._analyze_jpeg_quantization_from_bytes(file_bytes, results)

        except Exception as e:
            logger.error(f"Error analyzing JPEG quantization from file: {e}")
            results["image_analysis"]["jpeg_qtable_analysis"] = {
                "performed": False,
                "error": str(e),
                "suspicious": False,
            }

    def _analyze_jpeg_quantization_from_bytes(self, file_bytes: bytes, results: Dict[str, Any]) -> None:
        """Analyze JPEG quantization tables from bytes."""
        try:
            logger.debug("Analyzing JPEG quantization tables from bytes")

            # Look for quantization tables (marker FF DB)
            qtables_found = []
            standard_table_matches = []

            pos = 0
            data = file_bytes

            while pos < len(data) - 1:
                if data[pos] == 0xFF and data[pos + 1] == 0xDB:
                    # Found a quantization table
                    table_length = (data[pos + 2] << 8) + data[pos + 3]
                    table_data = data[pos + 4 : pos + 2 + table_length]
                    qtables_found.append(table_data.hex()[:20] + "...")

                    # Check if it matches standard tables
                    standard_match = len(table_data) == 64 or len(table_data) == 65
                    standard_table_matches.append(standard_match)

                    pos += 2 + table_length
                else:
                    pos += 1

            suspicious = len(qtables_found) > 2 or not all(standard_table_matches)

            result = {
                "performed": True,
                "qtables_found": len(qtables_found),
                "standard_tables": all(standard_table_matches),
                "suspicious": suspicious,
                "reason": (
                    "Multiple or non-standard quantization tables detected"
                    if suspicious
                    else "Normal quantization tables"
                ),
            }

            results["image_analysis"]["jpeg_qtable_analysis"] = result

            if suspicious:
                self._add_fraud_indicator(results, "JPEG quantization table analysis indicates possible manipulation")

            logger.info(f"JPEG quantization: {len(qtables_found)} tables, suspicious={suspicious}")

        except Exception as e:
            logger.error(f"Error analyzing JPEG quantization from bytes: {e}")
            results["image_analysis"]["jpeg_qtable_analysis"] = {
                "performed": False,
                "error": str(e),
                "suspicious": False,
            }

    def _perform_shadow_analysis(self, img_cv: np.ndarray, results: Dict[str, Any]) -> None:
        """Analyze shadow consistency."""
        try:
            logger.debug("Analyzing shadow consistency")

            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Threshold to find darker areas (potential shadows)
            _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out very small contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > 100]

            if len(significant_contours) < 2:
                logger.info(f"Shadow analysis: not enough shadows found ({len(significant_contours)})")
                results["image_analysis"]["shadow_consistency"] = {
                    "performed": True,
                    "shadow_count": len(significant_contours),
                    "suspicious": False,
                    "reason": "Not enough shadows for consistency analysis",
                }
                return

            # Calculate shadow directions based on gradient
            shadow_directions = []

            for contour in significant_contours:
                # Create mask for this shadow
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], 0, 255, -1)

                # Find gradient at shadow edges
                edge_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask

                # Apply Sobel operator to estimate gradient direction
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate direction angle
                edge_points = np.where(edge_mask > 0)
                if len(edge_points[0]) > 0:
                    avg_angle = np.arctan2(np.mean(sobely[edge_points]), np.mean(sobelx[edge_points])) * 180 / np.pi
                    shadow_directions.append(avg_angle)

            # Check consistency of directions
            if shadow_directions:
                direction_std = np.std(shadow_directions)

                # Get threshold from configuration
                shadow_direction_threshold = self.config.get("Thresholds", "shadow_direction_threshold", 45)

                suspicious = direction_std > shadow_direction_threshold

                result = {
                    "performed": True,
                    "shadow_count": len(significant_contours),
                    "direction_variance": float(direction_std),
                    "suspicious": suspicious,
                    "reason": (
                        "Inconsistent shadow directions detected"
                        if suspicious
                        else "Shadow directions appear consistent"
                    ),
                }

                if suspicious:
                    self._add_fraud_indicator(results, "Shadow analysis indicates possible inconsistencies in lighting")

                logger.info(
                    f"Shadow analysis: {len(significant_contours)} shadows, variance={direction_std:.2f}, suspicious={suspicious}"
                )
            else:
                result = {
                    "performed": True,
                    "shadow_count": 0,
                    "suspicious": False,
                    "reason": "No shadow directions could be analyzed",
                }

            results["image_analysis"]["shadow_consistency"] = result

        except Exception as e:
            logger.error(f"Error in shadow analysis: {e}")
            results["image_analysis"]["shadow_consistency"] = {"performed": False, "error": str(e), "suspicious": False}

    def _perform_cfa_analysis(self, img_cv: np.ndarray, results: Dict[str, Any]) -> None:
        """Analyze Color Filter Array interpolation patterns."""
        try:
            logger.debug("Analyzing CFA interpolation patterns")

            # Split channels
            b, g, r = cv2.split(img_cv)

            suspicious_blocks = 0
            total_blocks = 0
            block_size = 16
            h, w = img_cv.shape[:2]

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    total_blocks += 1

                    # Extract channel blocks
                    r_block = r[y : y + block_size, x : x + block_size]
                    g_block = g[y : y + block_size, x : x + block_size]
                    b_block = b[y : y + block_size, x : x + block_size]

                    # Calculate variances in different directions
                    r_var_h = np.var(np.diff(r_block, axis=1))
                    r_var_v = np.var(np.diff(r_block, axis=0))
                    g_var_h = np.var(np.diff(g_block, axis=1))
                    g_var_v = np.var(np.diff(g_block, axis=0))
                    b_var_h = np.var(np.diff(b_block, axis=1))
                    b_var_v = np.var(np.diff(b_block, axis=0))

                    # Check for unusual variance ratios
                    r_ratio = r_var_h / max(r_var_v, 0.0001)
                    g_ratio = g_var_h / max(g_var_v, 0.0001)
                    b_ratio = b_var_h / max(b_var_v, 0.0001)

                    # Check if patterns are consistent with typical CFA interpolation
                    if r_ratio > 3 or r_ratio < 0.33 or g_ratio > 3 or g_ratio < 0.33 or b_ratio > 3 or b_ratio < 0.33:
                        suspicious_blocks += 1

            suspicious_ratio = suspicious_blocks / max(total_blocks, 1)

            # Get threshold from configuration
            cfa_suspicious_ratio_threshold = self.config.get("Thresholds", "cfa_suspicious_ratio_threshold", 0.1)

            suspicious = suspicious_ratio > cfa_suspicious_ratio_threshold

            result = {
                "performed": True,
                "suspicious_blocks": suspicious_blocks,
                "total_blocks": total_blocks,
                "suspicious_ratio": float(suspicious_ratio),
                "suspicious": suspicious,
                "reason": (
                    "Inconsistent CFA interpolation patterns detected" if suspicious else "CFA patterns appear normal"
                ),
            }

            results["image_analysis"]["cfa_interpolation"] = result

            if suspicious:
                self._add_fraud_indicator(results, "CFA interpolation analysis indicates possible manipulation")

            logger.info(f"CFA analysis: ratio={suspicious_ratio:.3f}, suspicious={suspicious}")

        except Exception as e:
            logger.error(f"Error in CFA analysis: {e}")
            results["image_analysis"]["cfa_interpolation"] = {"performed": False, "error": str(e), "suspicious": False}

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
