"""Explicit domain exceptions."""


class NirmanNetraError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(NirmanNetraError):
    """Application configuration is invalid."""


class RasterReadError(NirmanNetraError):
    """A raster asset cannot be read."""


class RasterMetadataError(NirmanNetraError):
    """Raster metadata is missing or invalid."""


class InsufficientOverlapError(NirmanNetraError):
    """Raster footprints do not overlap enough for comparison."""


class RegistrationError(NirmanNetraError):
    """Visual registration cannot produce a reliable transform."""


class RegistrationQualityError(RegistrationError):
    """Input quality is too low for automatic registration."""


class TransformValidationError(RegistrationError):
    """An estimated transform is physically implausible."""


class CRSMismatchError(NirmanNetraError):
    """A CRS is invalid or incompatible with an operation."""


class GeometryValidationError(NirmanNetraError):
    """Geometry is empty, malformed, or topologically invalid."""


class StorageError(NirmanNetraError):
    """Object storage failed."""


class PersistenceError(NirmanNetraError):
    """Database persistence failed."""


class ExternalServiceError(NirmanNetraError):
    """An external service failed."""
