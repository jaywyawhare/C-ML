"""Logging and error-handling utilities."""

from cml._cml_lib import lib

LOG_SILENT = 0
LOG_ERROR = 1
LOG_WARN = 2
LOG_INFO = 3
LOG_DEBUG = 4


def set_log_level(level):
    lib.cml_set_log_level(level)


def has_error():
    return lib.cml_has_error()


def get_error():
    msg = lib.cml_get_error()
    if msg != lib.ffi.NULL:
        return lib.ffi.string(msg).decode("utf-8")
    return None


def clear_error():
    lib.cml_clear_error()
