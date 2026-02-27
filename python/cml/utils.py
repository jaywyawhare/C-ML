"""
Utility functions and utilities.
"""

from cml._cml_lib import lib

# Logging levels
LOG_SILENT = 0
LOG_ERROR = 1
LOG_WARN = 2
LOG_INFO = 3
LOG_DEBUG = 4


def set_log_level(level):
    """Set logging verbosity.

    Args:
        level: Logging level (0=SILENT, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG)

    Example:
        >>> set_log_level(LOG_DEBUG)  # Enable debug logging
    """
    lib.cml_set_log_level(level)


def has_error():
    """Check if an error occurred.

    Returns:
        True if error state is set

    Example:
        >>> if has_error():
        ...     print(get_error())
    """
    return lib.cml_has_error()


def get_error():
    """Get error message.

    Returns:
        Error message string or None

    Example:
        >>> try:
        ...     # CML operation
        ... except:
        ...     if has_error():
        ...         print(f"Error: {get_error()}")
    """
    msg = lib.cml_get_error()
    if msg != lib.ffi.NULL:
        return lib.ffi.string(msg).decode("utf-8")
    return None


def clear_error():
    """Clear error state.

    Example:
        >>> clear_error()  # Reset error flag
    """
    lib.cml_clear_error()
