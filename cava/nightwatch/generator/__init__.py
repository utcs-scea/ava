import logging

from nightwatch import (
    location,
    term,
    error,
    warning,
    info,
    ice,
    MultipleError,
    LocatedError,
    capture_errors,
    captured_errors,
    _build_assert,
    _build_requires,
    _build_expects,
)

__all__ = [
    "logger",
    "GenerateError",
    "MultipleError",
    "generate_assert",
    "generate_requires",
    "generate_expects",
    "term",
    "error",
    "warning",
    "info",
    "ice",
    "location",
    "capture_errors",
    "captured_errors",
]

logger = logging.getLogger(__name__)


class GenerateError(LocatedError):
    def __init__(self, *args, loc=None):
        super().__init__(*args, loc=loc, phase="generate")


generate_assert = _build_assert(GenerateError)
generate_requires = _build_requires(GenerateError)
generate_expects = _build_expects(GenerateError)
