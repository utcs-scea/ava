from functools import reduce
from typing import Callable, Set, Optional, Union

from nightwatch.generator import generate_assert

_known_constants = frozenset(
    [
        "NW_SYNC",
        "NW_ASYNC",
        "NW_FLUSH",
        "NW_NONE",
        "NW_HANDLE",
        "NW_OPAQUE",
        "NW_BUFFER",
        "NW_CALLBACK",
        "NW_CALLBACK_REGISTRATION",
        "NW_FILE",
        "NW_ZEROCOPY_BUFFER",
        "AVA_NONE",
        "AVA_COUPLED",
        "AVA_STATIC",
        "AVA_CALL",
        "AVA_MANUAL",
        "NULL",
        "malloc",
        "free",
        "ava_zerocopy_alloc",
        "ava_zerocopy_free",
    ]
)
_boolean_constants = frozenset([0, 1])


def _value_set_union(a: frozenset, b: frozenset) -> Optional[frozenset]:
    if a is None or b is None:
        return None
    return (a or frozenset()).union(b or frozenset())


def _value_set_isdisjoint(a: frozenset, b: frozenset) -> bool:
    if a is None or b is None:
        return False
    return a.isdisjoint(b)


def _value_set_issingltonsame(a: frozenset, b: frozenset) -> bool:
    if a is None or b is None:
        return False
    return a == b and len(a) == 1


def _parse_bool(v: str):
    if v.lower() in ("true", "1"):
        return 1
    if v.lower() in ("false", "0"):
        return 0
    raise ValueError("Boolean value expected.")


class _ExprMetaclass(type):
    def __call__(cls, code: Union["Expr", Callable, str, int], value_set: Optional[frozenset] = None):
        generate_assert(code is not None, "None is not valid in C expressions.")
        if isinstance(code, Expr):
            return code
        if callable(code):
            return Expr(code(), value_set)
        return super(_ExprMetaclass, cls).__call__(code, value_set)


class Expr(metaclass=_ExprMetaclass):
    value_set: Optional[frozenset]

    def __init__(self, code: Union[int, str], value_set: Optional[frozenset] = None) -> None:
        code = code.strip() if hasattr(code, "strip") and "#" not in code else code
        if isinstance(code, bool):
            code = int(code)
        self._code = code
        if value_set:
            self.value_set = frozenset(value_set)
        elif self.is_constant():
            self.value_set = frozenset([self.constant_value])
        else:
            self.value_set = None

    @property
    def code(self) -> str:
        return self._code

    def is_true(self) -> bool:
        return self.is_constant() and self.constant_value == 1

    def is_false(self) -> bool:
        return self.is_constant() and self.constant_value == 0

    @property
    def constant_value(self) -> Union[int, str]:
        if self._code in _known_constants:
            return self._code
        try:
            return int(str(self._code))
        except ValueError:
            try:
                return _parse_bool(str(self._code))
            except ValueError:
                # pylint: disable=raise-missing-from
                raise ValueError("CExpr is not a constant.")

    def is_constant(self, value: Optional[int] = None) -> bool:
        try:
            if value is None:
                # Get the constant value to trigger an exception (caught below) if it cannot be accessed.
                # pylint: disable=pointless-statement
                self.constant_value
                return True
            return self.constant_value == value
        except ValueError:
            return False

    def equals(self, other: str) -> "Expr":
        other = Expr(other)
        if _value_set_issingltonsame(self.value_set, other.value_set) or str(self) == str(other):
            return Expr(1)
        if (
            _value_set_isdisjoint(self.value_set, other.value_set)
            or self.is_constant()
            and other.is_constant()
            and self != other
        ):
            return Expr(0)
        return Expr(str(self.group()) + " == " + str(other.group()), value_set=_boolean_constants)

    def one_of(self, values: Set[str]) -> "Expr":
        if self.value_set and self.value_set.issubset(values):
            return Expr(1)
        return reduce(lambda accum, v: accum | self.equals(v), values, Expr(0))

    def not_equals(self, other: str) -> "Expr":
        other = Expr(other)
        if _value_set_issingltonsame(self.value_set, other.value_set) or str(self) == str(other):
            return Expr(0)
        if (
            _value_set_isdisjoint(self.value_set, other.value_set)
            or self.is_constant()
            and other.is_constant()
            and self != other
        ):
            return Expr(1)
        return Expr(str(self.group()) + " != " + str(other.group()), value_set=_boolean_constants)

    def __invert__(self) -> "Expr":
        if self.is_true():
            return Expr(0)
        if self.is_false():
            return Expr(1)
        return Expr("!" + str(self.group()), value_set=_boolean_constants)

    def __and__(self, other: Union[bool, str, "Expr"]) -> "Expr":
        other = Expr(other)
        if self.is_true() or other.is_false():
            return other
        if self.is_false() or other.is_true():
            return self
        return Expr(str(self) + " && " + str(other), value_set=_boolean_constants)

    def __or__(self, other: Union[bool, "Expr"]) -> "Expr":
        other = Expr(other)
        if self.is_false() or other.is_true():
            return other
        if self.is_true() or other.is_false():
            return self
        return Expr(str(self) + " || " + str(other), value_set=_boolean_constants)

    def __gt__(self, other: int) -> "Expr":
        other = Expr(other)
        if str(self) == str(other):
            return Expr(0)
        if self.is_constant() and other.is_constant():
            try:
                return Expr(self.constant_value > other.constant_value)
            except TypeError:
                # The types are not compatible in python. Emit a real C comparison.
                pass
        return Expr(str(self.group()) + " > " + str(other.group()), value_set=_boolean_constants)

    def __ge__(self, other):
        other = Expr(other)
        if str(self) == str(other):
            return Expr(0)
        if self.is_constant() and other.is_constant():
            try:
                return Expr(self.constant_value >= other.constant_value)
            except TypeError:
                # The types are not compatible in python. Emit a real C comparison.
                pass
        return Expr(str(self.group()) + " >= " + str(other.group()), value_set=_boolean_constants)

    def group(self) -> "Expr":
        return Expr("(" + str(self) + ")", value_set=self.value_set)

    def scope(self) -> "Expr":
        return Expr("{" + str(self) + "}") if str(self) else self

    def if_then_else_expression(
        self, then_branch: Union[int, str, "Expr"], else_branch: Union[int, str, "Expr"]
    ) -> "Expr":
        if self.is_true() or str(then_branch) == str(else_branch):
            then_branch = Expr(then_branch)
            return then_branch
        else_branch = Expr(else_branch)
        if self.is_false():
            return else_branch
        then_branch = Expr(then_branch)
        if then_branch.is_true() and else_branch.is_false():
            return self
        return Expr(
            str(self.group()) + " ? " + str(then_branch.group()) + " : " + str(else_branch.group()),
            value_set=_value_set_union(then_branch.value_set, else_branch.value_set),
        )

    def if_then_else(
        self, then_branch: Union[str, "Expr", Callable], else_branch: Union[str, "Expr", Callable] = ""
    ) -> "Expr":
        if self.is_true() or str(then_branch).strip() == str(else_branch).strip():
            then_branch = Expr(then_branch)
            return then_branch
        else_branch = Expr(else_branch)
        if self.is_false():
            return else_branch
        else_code = ""
        if else_branch:
            else_code = f"else {{ {else_branch} }}"
        then_branch = Expr(then_branch)
        if then_branch == else_branch:
            return then_branch
        return Expr(
            f"""
        if ({self}) {{ {then_branch} }} {else_code}
        """
        )

    def then(self, other: Union[str, "Expr"]) -> "Expr":
        return Expr(str(self) + str(other))

    def __eq__(self, other: Union[str, "Expr"]) -> bool:
        if hasattr(other, "code"):
            return self._code == other.code
        return self._code == other

    def __str__(self) -> str:
        code = self._code
        if isinstance(code, str) and "#" in code:
            code += "\n"
        return str(code)

    def __repr__(self):
        return f"`{self._code}`"

    def __hash__(self):
        return hash(self._code)

    def __bool__(self) -> bool:
        return bool(str(self._code))


ExprOrStr = Union[Expr, str]
