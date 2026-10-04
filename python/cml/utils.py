"""Logging and error-handling utilities."""

from cml._cml_lib import ffi, lib

# Log levels matching C enum LogLevel in logging.h
LOG_DEBUG = 0
LOG_INFO = 1
LOG_WARNING = 2
LOG_ERROR = 3


def set_log_level(level):
    """Set the CML log level.

    Args:
        level: One of LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR.
    """
    lib.cml_set_log_level(level)


def has_error():
    """Return whether the C error stack holds any unconsumed errors."""
    return lib.error_stack_has_errors()


def get_error():
    """Return the last C error message as a ``str``, or ``None`` if there is none."""
    msg = lib.error_stack_get_last_message()
    if msg == ffi.NULL:
        return None
    return ffi.string(msg).decode("utf-8")


def get_error_code():
    """Return the integer code of the most recent C error."""
    return lib.error_stack_get_last_code()


def clear_error():
    """Clear the last recorded C error."""
    lib.cml_clear_last_error()
