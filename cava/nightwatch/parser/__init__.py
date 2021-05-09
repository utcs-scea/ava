import logging

import nightwatch
from nightwatch import location, term, error, warning, info, ice, MultipleError, capture_errors, captured_errors

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


class ParseError(nightwatch.LocatedError):
    def __init__(self, *args, loc=None):
        super().__init__(*args, loc=loc, phase="parse")


parse_assert = nightwatch._build_assert(ParseError)
parse_requires = nightwatch._build_requires(ParseError)
parse_expects = nightwatch._build_expects(ParseError)
