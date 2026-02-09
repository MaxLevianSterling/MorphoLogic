# src/morphologic/exceptions.py
from __future__ import annotations

class MorphologicError(Exception):
    """Base for all domain errors."""

class ConfigError(MorphologicError):
    """Invalid or missing configuration."""

class DataNotFound(MorphologicError):
    """Required file(s) or directory not found."""

class SWCParseError(MorphologicError):
    """SWC file unreadable or invalid."""

class ImageIOError(MorphologicError):
    """Image file unreadable or invalid."""

class ValidationError(MorphologicError):
    """Input data fails semantic checks."""

class MetricComputationError(MorphologicError):
    """Metrics step failed."""

class VisualizationError(MorphologicError):
    """Plotting/figure export failed."""