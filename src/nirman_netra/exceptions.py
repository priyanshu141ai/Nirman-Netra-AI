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


class DatasetValidationError(NirmanNetraError):
    """A training dataset violates its schema or split policy."""


class TrainingError(NirmanNetraError):
    """Baseline training cannot complete under the requested configuration."""


class ArtifactValidationError(NirmanNetraError):
    """A versioned model artifact is missing, corrupt, or incompatible."""


class InferenceValidationError(NirmanNetraError):
    """Inference input does not match the model artifact schema."""


class CRSMismatchError(NirmanNetraError):
    """A CRS is invalid or incompatible with an operation."""


class GeometryValidationError(NirmanNetraError):
    """Geometry is empty, malformed, or topologically invalid."""


class StorageError(NirmanNetraError):
    """Object storage failed."""


class PersistenceError(NirmanNetraError):
    """Database persistence failed."""


class CaseTransitionError(NirmanNetraError):
    """A case lifecycle transition is invalid."""


class AuthorizationError(NirmanNetraError):
    """An actor role is not allowed to perform an operation."""


class EvidenceIntegrityError(NirmanNetraError):
    """Evidence content or chain-of-custody metadata is invalid."""


class AuditIntegrityError(NirmanNetraError):
    """An append-only timeline or audit chain is invalid."""


class ModelNotAvailableError(NirmanNetraError):
    """A specifically requested approved model artifact is unavailable."""


class ModelCompatibilityError(NirmanNetraError):
    """A model artifact is valid but incompatible with the requested operation."""


class ModelChecksumMismatchError(ModelCompatibilityError):
    """A requested model file does not match its recorded checksum."""


class ModelSchemaMismatchError(ModelCompatibilityError):
    """A requested model artifact violates the required integration schema."""


class RuleSetNotAvailableError(NirmanNetraError):
    """A specifically requested municipal rule set is unavailable or incompatible."""


class QueueUnavailableError(NirmanNetraError):
    """The configured processing queue cannot accept or run work."""


class JobStateError(NirmanNetraError):
    """A processing job operation is invalid for its current state."""


class RecordNotFoundError(NirmanNetraError):
    """A requested application record does not exist."""


class UploadValidationError(NirmanNetraError):
    """Uploaded asset metadata or object reference is invalid."""


class ExternalServiceError(NirmanNetraError):
    """An external service failed."""
