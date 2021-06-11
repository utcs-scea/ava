import logging

from nightwatch import (
    location,
    term,
    error,
    warning,
    info,
    ice,
    LocatedError,
    MultipleError,
    capture_errors,
    captured_errors,
    _build_assert,
    _build_requires,
    _build_expects,
)

__all__ = [
    "logger",
    "ParseError",
    "MultipleError",
    "parse_assert",
    "parse_requires",
    "parse_expects",
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


class ParseError(LocatedError):
    def __init__(self, *args, loc=None):
        super().__init__(*args, loc=loc, phase="parse")


parse_assert = _build_assert(ParseError)
parse_requires = _build_requires(ParseError)
parse_expects = _build_expects(ParseError)
