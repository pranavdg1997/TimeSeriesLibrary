"""Custom exceptions for universal_ts library."""


class UniversalTSError(Exception):
    """Base exception for universal_ts library."""
    pass


class DataValidationError(UniversalTSError):
    """Raised when input data fails validation."""
    pass


class BackendNotFoundError(UniversalTSError):
    """Raised when requested backend is not available."""
    pass


class BackendNotInstalledError(UniversalTSError):
    """Raised when backend dependencies are not installed."""
    pass


class UnsupportedOperationError(UniversalTSError):
    """Raised when an operation is not supported by the backend."""
    pass


class FitNotCalledError(UniversalTSError):
    """Raised when predict is called before fit."""
    pass
