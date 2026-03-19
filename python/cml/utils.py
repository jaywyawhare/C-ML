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
    return lib.error_stack_has_errors()


def get_error():
    msg = lib.error_stack_get_last_message()
    if msg == ffi.NULL:
        return None
    return ffi.string(msg).decode("utf-8")


def get_error_code():
    return lib.error_stack_get_last_code()


def clear_error():
    lib.error_stack_cleanup()
    lib.error_stack_init()
