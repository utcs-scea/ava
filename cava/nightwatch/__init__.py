import logging
import re
import sys
import threading
from contextlib import contextmanager
import blessings


__all__ = [
    "NightWatchError",
    "MultipleError",
    "term",
    "error",
    "warning",
    "info",
    "ice",
    "LocatedError",
    "location",
    "capture_errors",
    "captured_errors",
]

logger = logging.getLogger(__name__)


class NightWatchError(Exception):
    def __init__(self, *args, phase=None):
        super().__init__(*args)
        self.phase = f" ({phase})" if phase else ""
        self.reported = False

    def report(self):
        if not self.reported:
            print(self.pretty, file=sys.stderr)
            self.reported = True

    @property
    def pretty(self):
        return f"""{self.phase}: {self}"""


class MultipleError(Exception):
    def __new__(cls, *exceptions):
        if len(exceptions) > 1:
            return super().__new__(cls, *exceptions)
        if exceptions:
            return exceptions[0]
        raise ValueError("MultipleError should have multiple exceptions as its arguments.")

    def report(self):
        for exc in self.args:
            if hasattr(exc, "reported") and not exc.reported:
                exc.report()

    def __str__(self):
        return "; ".join(str(e) for e in self.args)

    def __repr__(self):
        return "MultipleError(" + ", ".join(repr(e) for e in self.args) + ")"


try:
    term = blessings.Terminal(stream=sys.stderr)
# pylint: disable=bare-except
except:
    term = blessings.Terminal(stream=sys.stderr, force_styling=None)

ansi_regex = (
    r"\x1b("
    r"(\[\??\d+[hl])|"
    r"([=<>a-kzNM78])|"
    r"([\(\)][a-b0-2])|"
    r"(\[\d{0,3}[ma-dgkjqi])|"
    r"(\[\d+;\d+[hfy]?)|"
    r"(\[;?[hf])|"
    r"(#[3-68])|"
    r"([01356]n)|"
    r"(O[mlnp-z]?)|"
    r"(/Z)|"
    r"(\d+)|"
    r"(\[\?\d;\d0c)|"
    r"(\d;\dR))"
)
ansi_escape = re.compile(ansi_regex, flags=re.IGNORECASE)


def strip_color(s):
    return ansi_escape.sub("", s)


error = term.bright_red("ERROR")
warning = term.bright_magenta("Warning")
info = term.bright_cyan("info")
ice = term.black_on_bright_red("Meatballs?")


class LocatedError(NightWatchError):
    def __init__(self, *args, loc=None, phase=None):
        super().__init__(*args, phase=phase)
        self.loc = loc or ""
        self.reported = False

    def __str__(self):
        return str(self.loc) + ": " + ": ".join(strip_color(str(v)) for v in self.args)

    def __repr__(self):
        return str(self.loc) + ": " + ": ".join(repr(v) for v in self.args)

    @property
    def pretty(self):
        indent = "  "
        descriptions = ":\n".join(indent + (v.pretty if hasattr(v, "pretty") else str(v)) for v in self.args[:-1])
        # logger.exception(self)
        return f"""\
{term.bold_white(str(self.loc))}:{self.phase}
{descriptions}:
{indent}{indent}{self.args[-1]}
""".strip()

    # def improve(self, description, loc=None):
    #     if not self.loc and loc:
    #         self.loc = loc
    #     self.args = (description,) + self.args


_parse_state = threading.local()
_parse_state.descriptions = []
_parse_state.locations = [None]
_parse_state.errors = [None]


def _build_assert(ErrorType):
    def _assert(pred, *description, loc=None, kind=ice):
        if not pred:
            raise ErrorType(
                kind, *(_parse_state.descriptions + list(description)), loc=loc or _parse_state.locations[-1]
            )

    return _assert


def _build_requires(ErrorType):
    def _requires(pred, *description, loc=None, kind=error):
        if not pred:
            raise ErrorType(
                kind, *(_parse_state.descriptions + list(description)), loc=loc or _parse_state.locations[-1]
            )

    return _requires


def _build_expects(ErrorType):
    def _expects(pred, *description, loc=None, kind=warning):
        if not pred:
            ErrorType(
                kind, *(_parse_state.descriptions + list(description)), loc=loc or _parse_state.locations[-1]
            ).report()

    return _expects


@contextmanager
def location(description, loc=None, report_continue=None):
    if description:
        _parse_state.descriptions.append(description)
    if loc:
        _parse_state.locations.append(loc)
    try:
        yield
    except LocatedError as e:
        # e.improve(description, loc)
        if report_continue is None:
            raise e
        if hasattr(report_continue, "append"):
            e.report()
            report_continue.append(e)
        assert False
    finally:
        if description:
            _parse_state.descriptions.pop()
        if loc:
            _parse_state.locations.pop()


@contextmanager
def capture_errors():
    _parse_state.errors.append([])
    try:
        yield
    except LocatedError as e:
        e.report()
        _parse_state.errors[-1].append(e)
    finally:
        errors = _parse_state.errors.pop()
    return errors


def captured_errors():
    return MultipleError(*_parse_state.errors[-1])
