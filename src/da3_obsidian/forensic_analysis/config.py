"""
Configuration management for forensic analysis.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ForensicConfig:
    """Configuration settings for forensic analysis."""

    # General settings
    log_level: str = "INFO"
    max_pdf_pages: int = 20
    max_images: int = 50
    jpeg_quality: int = 95

    # Analysis thresholds
    ela_mean_threshold: float = 5.0
    ela_std_threshold: float = 10.0
    noise_mean_threshold: float = 2.0
    noise_std_threshold: float = 3.0
    copy_move_threshold: int = 10
    ghost_ratio_threshold: float = 0.05
    shadow_direction_threshold: float = 45.0
    straight_lines_threshold: int = 5
    discontinuity_ratio_threshold: float = 0.2
    cfa_suspicious_ratio_threshold: float = 0.1
    font_count_threshold: int = 3

    def __post_init__(self):
        """Initialize section mapping after dataclass creation."""
        # Map nested section.key access to flat dataclass attributes
        self._section_mapping = {
            "General": {"log_level": "log_level"},
            "Analysis": {"max_pdf_pages": "max_pdf_pages", "max_images": "max_images", "jpeg_quality": "jpeg_quality"},
            "Thresholds": {
                "ela_mean_threshold": "ela_mean_threshold",
                "ela_std_threshold": "ela_std_threshold",
                "noise_mean_threshold": "noise_mean_threshold",
                "noise_std_threshold": "noise_std_threshold",
                "copy_move_threshold": "copy_move_threshold",
                "ghost_ratio_threshold": "ghost_ratio_threshold",
                "shadow_direction_threshold": "shadow_direction_threshold",
                "straight_lines_threshold": "straight_lines_threshold",
                "discontinuity_ratio_threshold": "discontinuity_ratio_threshold",
                "cfa_suspicious_ratio_threshold": "cfa_suspicious_ratio_threshold",
                "font_count_threshold": "font_count_threshold",
            },
        }

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value using nested section.key access.

        Args:
            section: Configuration section name
            key: Configuration key name
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        if section not in self._section_mapping:
            logger.warning(f"Unknown config section: {section}")
            return default

        if key not in self._section_mapping[section]:
            logger.warning(f"Unknown config key: {section}.{key}")
            return default

        attr_name = self._section_mapping[section][key]
        return getattr(self, attr_name, default)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ForensicConfig":
        """
        Create configuration from nested dictionary.

        Args:
            config_dict: Nested dictionary with configuration values

        Returns:
            ForensicConfig instance
        """
        # Flatten nested dictionary to match dataclass fields
        flattened = {}

        # Handle flat dictionary (for backward compatibility)
        if any(key in cls.__dataclass_fields__ for key in config_dict.keys()):
            flattened = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        else:
            # Handle nested dictionary
            section_mapping = {
                "General": {"log_level": "log_level"},
                "Analysis": {
                    "max_pdf_pages": "max_pdf_pages",
                    "max_images": "max_images",
                    "jpeg_quality": "jpeg_quality",
                },
                "Thresholds": {
                    "ela_mean_threshold": "ela_mean_threshold",
                    "ela_std_threshold": "ela_std_threshold",
                    "noise_mean_threshold": "noise_mean_threshold",
                    "noise_std_threshold": "noise_std_threshold",
                    "copy_move_threshold": "copy_move_threshold",
                    "ghost_ratio_threshold": "ghost_ratio_threshold",
                    "shadow_direction_threshold": "shadow_direction_threshold",
                    "straight_lines_threshold": "straight_lines_threshold",
                    "discontinuity_ratio_threshold": "discontinuity_ratio_threshold",
                    "cfa_suspicious_ratio_threshold": "cfa_suspicious_ratio_threshold",
                    "font_count_threshold": "font_count_threshold",
                },
            }

            for section_name, section_data in config_dict.items():
                if section_name in section_mapping:
                    for key, value in section_data.items():
                        if key in section_mapping[section_name]:
                            attr_name = section_mapping[section_name][key]
                            flattened[attr_name] = value

        return cls(**flattened)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary."""
        return {
            "General": {"log_level": self.log_level},
            "Analysis": {
                "max_pdf_pages": self.max_pdf_pages,
                "max_images": self.max_images,
                "jpeg_quality": self.jpeg_quality,
            },
            "Thresholds": {
                "ela_mean_threshold": self.ela_mean_threshold,
                "ela_std_threshold": self.ela_std_threshold,
                "noise_mean_threshold": self.noise_mean_threshold,
                "noise_std_threshold": self.noise_std_threshold,
                "copy_move_threshold": self.copy_move_threshold,
                "ghost_ratio_threshold": self.ghost_ratio_threshold,
                "shadow_direction_threshold": self.shadow_direction_threshold,
                "straight_lines_threshold": self.straight_lines_threshold,
                "discontinuity_ratio_threshold": self.discontinuity_ratio_threshold,
                "cfa_suspicious_ratio_threshold": self.cfa_suspicious_ratio_threshold,
                "font_count_threshold": self.font_count_threshold,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate ranges
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 0 and 100")

        if self.max_pdf_pages < 1:
            raise ValueError("max_pdf_pages must be at least 1")

        if self.max_images < 1:
            raise ValueError("max_images must be at least 1")

        # Validate thresholds are positive
        threshold_fields = [
            "ela_mean_threshold",
            "ela_std_threshold",
            "noise_mean_threshold",
            "noise_std_threshold",
            "ghost_ratio_threshold",
            "shadow_direction_threshold",
            "discontinuity_ratio_threshold",
            "cfa_suspicious_ratio_threshold",
        ]

        for field in threshold_fields:
            value = getattr(self, field)
            if value < 0:
                raise ValueError(f"{field} must be non-negative")

        # Validate integer thresholds
        if self.copy_move_threshold < 0:
            raise ValueError("copy_move_threshold must be non-negative")

        if self.straight_lines_threshold < 0:
            raise ValueError("straight_lines_threshold must be non-negative")

        if self.font_count_threshold < 1:
            raise ValueError("font_count_threshold must be at least 1")

        return True

    def get_analysis_limits(self) -> Dict[str, int]:
        """Get analysis limits for processing."""
        return {"max_pdf_pages": self.max_pdf_pages, "max_images": self.max_images}

    def get_thresholds(self) -> Dict[str, float]:
        """Get all threshold values."""
        return {
            "ela_mean_threshold": self.ela_mean_threshold,
            "ela_std_threshold": self.ela_std_threshold,
            "noise_mean_threshold": self.noise_mean_threshold,
            "noise_std_threshold": self.noise_std_threshold,
            "copy_move_threshold": float(self.copy_move_threshold),
            "ghost_ratio_threshold": self.ghost_ratio_threshold,
            "shadow_direction_threshold": self.shadow_direction_threshold,
            "straight_lines_threshold": float(self.straight_lines_threshold),
            "discontinuity_ratio_threshold": self.discontinuity_ratio_threshold,
            "cfa_suspicious_ratio_threshold": self.cfa_suspicious_ratio_threshold,
            "font_count_threshold": float(self.font_count_threshold),
        }
