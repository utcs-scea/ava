import logging

import nightwatch
from nightwatch import location, term, error, warning, info, ice, MultipleError, capture_errors, captured_errors

__all__ = ["logger",
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
           "captured_errors"]

logger = logging.getLogger(__name__)


class GenerateError(nightwatch.LocatedError):
    def __init__(self, *args, loc=None):
        super().__init__(*args, loc=loc, phase="generate")


generate_assert = nightwatch._build_assert(GenerateError)
generate_requires = nightwatch._build_requires(GenerateError)
generate_expects = nightwatch._build_expects(GenerateError)
