"""
Image OCR processing module using Tesseract with advanced preprocessing.
"""

import logging
import math

import cv2
import imutils
import numpy as np
import pytesseract
from deskew import determine_skew

from .quality_checker import QualityChecker

logger = logging.getLogger(__name__)

# Tesseract language mappings
TESSERACT_LANG_MAPPING = {"en": "eng", "es": "spa", "fr": "fra", "it": "ita"}


class ImageProcessor:
    """Handles image OCR with advanced preprocessing and quality validation."""

    def __init__(self, language: str = "en"):
        """
        Initialize image processor.

        Args:
            language: Language code for OCR ('en', 'es', 'fr', 'it')
        """
        self.language = language
        self.tesseract_lang = TESSERACT_LANG_MAPPING.get(language, "eng")
        self.quality_checker = QualityChecker()

    def read_image(self, path: str) -> np.ndarray:
        """Load image from disk in grayscale."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {path}")
        logger.info(f"Image loaded successfully from {path}")
        return img

    def dynamic_resize_image(self, img: np.ndarray, max_dimension: int = 4096) -> np.ndarray:
        """Dynamically resize image if larger than max_dimension."""
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or not valid")

        height, width = img.shape[:2]
        if max(width, height) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            logger.debug(f"Image resized to ({new_width}, {new_height})")
            return resized_img

        return img

    def rotate_image(self, img: np.ndarray, angle: float, background=(0, 0, 0)) -> np.ndarray:
        """Rotate image by given angle."""
        old_width, old_height = img.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2

        img = cv2.warpAffine(img, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
        logger.debug(f"Image rotated by {angle} degrees")
        return img

    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew image by detecting and correcting skew angle."""
        try:
            angle = determine_skew(img)
            if angle is not None:
                logger.debug(f"Deskewing with angle: {angle:.2f} degrees")
                img = self.rotate_image(img, -angle)
            return img
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return img

    def apply_transformations(self, img: np.ndarray) -> np.ndarray:
        """Apply morphological operations and contour filtering."""
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or not valid")

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        filtered_contours = [c for c in contours if cv2.contourArea(c) >= 15]

        if not filtered_contours:
            logger.warning("No valid contours found")
            return img

        try:
            stacked_contours = np.vstack(filtered_contours)
            convex_hull = cv2.convexHull(stacked_contours)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [convex_hull], -1, 255, -1)
            mask = cv2.dilate(mask, None, iterations=2)
            img = cv2.bitwise_and(img, img, mask=mask)
        except ValueError:
            logger.warning("Error stacking contours, returning original image")

        return img

    def apply_clahe(self, img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(3, 3)) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(img)
        logger.debug(f"CLAHE applied with clip limit {clip_limit}")
        return enhanced_img

    def median_blur(self, img: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Apply median blur to reduce noise."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")

        blurred_img = cv2.medianBlur(img, ksize)
        logger.debug(f"Median blur applied with kernel size {ksize}")
        return blurred_img

    def otsu_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding for binarization."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")

        _, thresholded_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug("Otsu's thresholding applied")
        return thresholded_img

    def bilateral_filter(self, img: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
        """Apply bilateral filtering to reduce noise while preserving edges."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")

        filtered_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        logger.debug(f"Bilateral filter applied with d={d}")
        return filtered_img

    def adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")

        thresholded_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        logger.debug("Adaptive thresholding applied")
        return thresholded_img

    def morphological_transform(self, img: np.ndarray, kernel_size=(3, 3)) -> np.ndarray:
        """Apply morphological operations to clean up the image."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.dilate(img, kernel, iterations=1)
        logger.debug(f"Morphological transformation applied with kernel {kernel_size}")
        return img

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Apply complete preprocessing pipeline to image."""
        try:
            # Ensure grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = self.dynamic_resize_image(img)
            img = self.deskew_image(img)
            img = self.median_blur(img)
            img = self.apply_clahe(img)
            img = self.bilateral_filter(img)
            img = self.adaptive_threshold(img)
            img = self.apply_transformations(img)
            img = self.morphological_transform(img)

            logger.debug("Image preprocessing completed successfully")
            return img

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            raise

    def extract_from_array(self, img: np.ndarray) -> str:
        """Extract text from numpy image array using OCR."""
        try:
            processed_img = self.preprocess_image(img)

            config_base = "--oem 3"
            text = pytesseract.image_to_string(processed_img, lang=self.tesseract_lang, config=f"--psm 6 {config_base}")

            logger.debug(f"OCR extraction completed, {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Failed to extract text from image array: {e}")
            raise

    def extract_from_file(self, image_path: str) -> str:
        """Extract text from image file using OCR."""
        img = self.read_image(image_path)
        return self.extract_from_array(img)
