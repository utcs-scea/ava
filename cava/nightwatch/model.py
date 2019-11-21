from collections import namedtuple
import re
from copy import copy
from typing import List, Set, Iterator, Collection, Optional, Union, Mapping

from toposort import toposort_flatten, CircularDependencyError

from nightwatch import c_dsl
from nightwatch.annotation_set import default_annotations, Conditional
from nightwatch.c_dsl import Expr, ExprOrStr
from .indent import indent_c
from .parser import parse_assert, parse_requires, parse_expects

_annotation_prefix = "ava_"
ASCRIBE_TYPES = False


def lines(strs):
    return "\n".join(s for s in strs if s)


def uncamel(self):
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", self)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def guard_macro_spelling(self):
    s1 = re.sub(r"[^a-zA-Z0-9]", r"_", self)
    return "__" + s1.upper() + "__"


def identifier_spelling(self):
    """Take any string and build an identifier from it."""
    s1 = re.sub(r"\*", r"P", self)
    s2 = re.sub(r"[^a-zA-Z0-9]", r"_", s1)
    return s2


def flag(flag, pred, sep=" "):
    return flag + sep if pred else ""


def _clean_type_string(s):
    return s.replace(" ", "")


def _type_strings_equal(t, u):
    return _clean_type_string(t) == _clean_type_string(u)


NIGHTWATCH_PREFIX = "__NIGHTWATCH_"

buffer_index_spelling = "ava_index"


# Location

class Location(namedtuple("Location", ["filename", "line", "column", "offset"])):
    __slots__ = ()

    def __str__(self):
        ss = []
        if self.filename:
            ss.append(self.filename)
        if isinstance(self.line, int) or self.line:
            ss.append(str(self.line))
        if isinstance(self.column, int) or self.column:
            ss.append(str(self.column))
        return ":".join(ss)


# Types

class Type(object):
    success: Optional[ExprOrStr]
    transfer: Optional[ExprOrStr]
    spelling: str
    pointee: Optional["Type"]
    fields: Mapping[str, "Type"]
    buffer: Optional[ExprOrStr]
    lifetime: Optional[ExprOrStr]
    lifetime_coupled: Optional[ExprOrStr]
    buffer_allocator: ExprOrStr
    buffer_deallocator: ExprOrStr

    def __init__(self, spelling, **annotations):
        self.fields = {}
        self.spelling = spelling
        self.success = None
        self.allocates_resources = {}
        self.deallocates_resources = {}
        self.type_cast = None
        self.transfer = None
        self.lifetime = None
        self.lifetime_coupled = None
        self.original_type = self
        self.__dict__.update(annotations)
        parse_assert(self.transfer is not None, "The parser must set transfer.")
        parse_assert(self.spelling, "Types must have spellings.")
        parse_assert(self.type_cast is None or isinstance(self.type_cast, (str, Conditional)),
                     f"type_cast must be None, str, or Conditional: {self.type_cast}")
        parse_requires(not self.buffer or hasattr(self, "pointee"), "Buffer values must be of a pointer type")
        parse_requires(not (self.buffer_allocator != "malloc" or self.buffer_deallocator != "free") or self.buffer,
                       "Buffer (de)allocators are only allowed on buffers. (You probably forgot to add `ava_buffer(size)`.)")
        parse_assert(not (hasattr(self, "pointee") and self.fields),
                     "Types must be either a pointer or a struct, not both")

    @property
    def contained_types(self) -> Set:
        return {self}

    @property
    def is_void(self) -> bool:
        return self.spelling == "void" or self.spelling == "const void"

    @staticmethod
    def _drop_const(s: str) -> str:
        if s.startswith("const "):
            s = s[6:]
        if s.endswith("const"):
            s = s[:-5]
        return s

    @property
    def nonconst(self):
        new_spelling = self._drop_const(self.spelling)
        v = copy(self)
        v.spelling = new_spelling
        return v

    transfer_spellings = {
        "NW_BUFFER": None,
        "NW_OPAQUE": "opaque",
        "NW_FILE": "file",
        "NW_HANDLE": "handle",
        "NW_CALLBACK": "callback",
        "NW_CALLBACK_REGISTRATION": "callback_registration",
    }

    lifetime_spellings = {
        "AVA_CALL": None,
        "AVA_COUPLED": None,
        "AVA_STATIC": "static",
        "AVA_MANUAL": "manual",
    }

    hidden_annotations = {"location", "spelling", "buffer_deallocator", "callback_stub_function"}

    @property
    def annotations(self) -> str:
        annotations = ""
        late_annotations = ""
        for name, value in self.__dict__.items():
            if name in self.hidden_annotations:
                pass
            elif name == "pointee":
                anns = self.pointee.annotations
                if self.transfer not in ("NW_HANDLE", "NW_OPAQUE") and anns:
                    late_annotations += f"{_annotation_prefix}element {{ {anns} }}\n"
            elif name == "fields":
                for fname, field in self.fields.items():
                    anns = field.annotations
                    if anns:
                        late_annotations += f"{_annotation_prefix}field({fname}) {{ {anns} }}\n"
            elif name == "transfer" and value in self.transfer_spellings and value != default_annotations.get(name):
                if value in ("NW_CALLBACK", "NW_CALLBACK_REGISTRATION"):
                    annotations += f"{_annotation_prefix}{self.transfer_spellings[value]}({self.callback_stub_function});\n"
                elif self.transfer_spellings[value]:
                    annotations += f"{_annotation_prefix}{self.transfer_spellings[value]};\n"
            elif name == "lifetime" and value in self.lifetime_spellings and value != default_annotations.get(name):
                if self.lifetime_spellings[value]:
                    annotations += f"{_annotation_prefix}lifetime_{self.lifetime_spellings[value]};\n"
            elif name == "buffer":
                if self.transfer == "NW_BUFFER":
                    annotations += f"{_annotation_prefix}{name}({value});\n"
            elif name == "buffer_allocator" and value != default_annotations.get(name):
                annotations += f"{_annotation_prefix}{name}({value}, {self.buffer_deallocator});\n"
            elif name.endswith("allocates_resources") and value:
                for resource, amount in value.items():
                    annotations += f"{_annotation_prefix}{name}({resource}, {amount});\n"
            elif not name.startswith("_") and value != default_annotations.get(name):
                if isinstance(value, bool):
                    if value:
                        annotations += f"{_annotation_prefix}{name};\n"
                else:
                    if value:
                        annotations += f"{_annotation_prefix}{name}({value});\n"
        annotations += late_annotations
        return annotations

    def __str__(self):
        return self.spelling

    def __repr__(self):
        nl = "\n"
        return f"""<{self.spelling} {self.annotations.replace(nl, " ")}>"""

    def attach_to(self, name, additional_inner_type_elements="") -> str:
        s = self.spelling
        return f"{s}{additional_inner_type_elements} {name}"

    def spelled_with(self, additional_inner_type_elements="") -> str:
        return self.attach_to("", additional_inner_type_elements=additional_inner_type_elements)

    def ascribe_type(self, v, additional_inner_type_elements="") -> str:
        if ASCRIBE_TYPES:
            return f"__ava_check_type({self.spelled_with(additional_inner_type_elements)}, {v})"
        else:
            return v

    def cast_type(self, v, additional_inner_type_elements="") -> str:
        return f"({self.spelled_with(additional_inner_type_elements)})({v})"


class ConditionalType(Type):
    """
    A type-level if-statement specifying a type which varies based on a runtime expression.
    """
    predicate: ExprOrStr

    def __init__(self, predicate, then_type: Type, else_type: Type, original_type: Type):
        d = original_type.__dict__.copy()
        d.pop("spelling")
        super().__init__(original_type.spelling, **d)
        self.predicate = predicate
        self.then_type = then_type
        self.else_type = else_type
        self.original_type = original_type

    @property
    def contained_types(self):
        return self.then_type.contained_types | self.else_type.contained_types

    def attach_to(self, name, additional_inner_type_elements=""):
        return self.original_type.attach_to(name, additional_inner_type_elements)


class StaticArray(Type):
    """
    Fixed size array.
    """
    pointee: Type

    def __init__(self, spelling, **annotations):
        super().__init__(spelling, **annotations)
        assert self.buffer
        assert self.pointee

    def attach_to(self, name, additional_inner_type_elements=""):
        return self.pointee.attach_to(name, additional_inner_type_elements=additional_inner_type_elements+"*")


class FunctionPointer(Type):
    """
    A pointer to a function.
    """

    hidden_annotations = Type.hidden_annotations | {"argument_types", "return_type"}

    def __init__(self, spelling, pointee, return_type, argument_types, **annotations):
        args = ", ".join(str(a) for a in argument_types)
        spelling = f"{return_type} (*)({args})"
        super().__init__(spelling, **annotations)
        parse_assert(str(self.transfer) in ("NW_OPAQUE", "NW_CALLBACK", "NW_CALLBACK_REGISTRATION"),
                     "Function pointers must be opaque: " + str(self.transfer))
        self.return_type = return_type
        self.argument_types = argument_types

    @property
    def nonconst(self):
        v = super(FunctionPointer, self).nonconst
        v.return_type = v.return_type.nonconst
        return v

    def attach_to(self, name, additional_inner_type_elements=""):
        args = ", ".join(str(a) for a in self.argument_types)
        return f"{self.return_type} (*{additional_inner_type_elements}{name}) ({args})"

    def ascribe_type(self, v, additional_inner_type_elements="") -> str:
        # Do not perform ascription because it's hard for function types.
        return v


# Argument

RET_ARGUMENT_NAME = "ret"


class Argument(object):
    type: Type

    def __init__(self, name, type: Type, **annotations):
        assert isinstance(type, Type)
        self.name = name
        self.type = type
        self.depends_on = None
        self.implicit_argument = None
        self.value = None
        self.input = 0
        self.output = 0
        self.no_copy = False
        self.ret = False
        self.__dict__.update(annotations)

        parse_requires(not self.userdata or self.type.spelling == "void *",
                       "Type of userdata arguments must be exactly void*.")
        parse_requires(not self.implicit_argument or self.value,
                       f"Implicit arguments must have a value assigned to them: {self}")
        parse_requires(not any(t.buffer for t in self.contained_types) or (self.no_copy or self.input or self.output),
                      f"Arguments containing buffers must be either ava_input or "
                      f"ava_output (and it was not guessed). If you want no copies "
                      f"at all, provide ava_no_copy.")


    @property
    def contained_types(self) -> Set[Type]:
        """Return an iterable contains all types in this argument."""
        return self.type.contained_types

    def __str__(self):
        value_str = " = " + str(self.value) if self.value else ""
        return self.type.attach_to(self.name + value_str)

    def __lt__(self, other):
        a = self._all_arguments.index(self)
        b = self._all_arguments.index(other)
        return a < b

    hidden_annotations = {"ret", "function", "location", "name"}

    @property
    def annotations(self) -> str:
        annotations = ""
        for name, value in self.__dict__.items():
            if name in self.hidden_annotations:
                pass
            elif name == "type":
                pass # Handled below
            elif name == "depends_on" and value:
                annotations += f"{_annotation_prefix}{name}({', '.join(value)});\n"
            elif not name.startswith("_") and value != default_annotations.get(name):
                if isinstance(value, bool):
                    if value:
                        annotations += f"{_annotation_prefix}{name};\n"
                else:
                    if value:
                        annotations += f"{_annotation_prefix}{name}({value});\n"
        annotations += self.type.annotations
        if annotations:
            if self.ret:
                return f"{_annotation_prefix}return_value {{ {annotations} }}"
            else:
                return f"{_annotation_prefix}argument({self.name}) {{ {annotations} }}"
        else:
            return ""

    @property
    def declaration(self):
        return self.type.attach_to(self.name)

    @property
    def original_declaration(self):
        return self.type.original_type.attach_to(self.name)


# Function

class Function(object):
    name: str
    return_value: Argument
    _arguments: List[Argument]
    supported: bool
    callback_decl: bool
    ignore: bool
    generate_timing_code: bool
    disable_native: bool

    def __init__(self, name: str, return_value: Argument, arguments: List[Argument], location, **annotations):
        self.prologue = ""
        self.epilogue = ""
        self.logue_declarations = []
        self.name = name
        self.return_value = return_value
        self._original_arguments = arguments
        self.arguments = self._order_arguments(arguments, location)
        self.synchrony = "NW_SYNC"
        self.ignore = False
        self.callback_decl = False
        self.consumes_resources = {}
        self.supported = True
        self.location = location
        self.generate_timing_code = False
        self.disable_native = False
        self.__dict__.update(annotations)

        assert not self.callback_decl or hasattr(self, "type") and self.type

        self.return_value.function = self
        self.return_value.ret = True
        for a in arguments:
            a.function = self

        parse_assert(all(sum(1 for b in self.arguments if a.name == b.name) == 1 for a in self.arguments),
                     "All argument names much be different.")

    @property
    def real_arguments(self) -> Iterator[Argument]:
        return (a for a in self._original_arguments if not a.implicit_argument)

    @property
    def implicit_arguments(self) -> Iterator[Argument]:
        return (a for a in self._original_arguments if a.implicit_argument)

    @classmethod
    def _get_argument_by_name(cls, arguments, name: str, default = None) -> Argument:
        for a in arguments:
            if a.name == name:
                return a
        if default is not None:
            return default
        else:
            raise LookupError(name)

    @classmethod
    def _order_arguments(cls, arguments: List[Argument], location) -> List[Argument]:
        dag = {}
        # Build dag to have deps specified by depends_on, and a dep
        # chain through all the NON-depends_on arguments in order.
        for arg in arguments:
            arg._all_arguments = arguments
            try:
                if arg.depends_on:
                    dag[arg] = set(cls._get_argument_by_name(arguments, n) for n in arg.depends_on)
                else:
                    dag[arg] = set()
            except LookupError as e:
                parse_requires(False,
                               f"Unknown argument name: {e.args[0]}",
                               loc=arg.location)

        # Compute an order which honors the deps
        try:
            return toposort_flatten(dag, sort=True)
        except CircularDependencyError:
            parse_requires(False,
                           "The dependencies between arguments are cyclic.",
                           loc=location)

    @property
    def contained_types(self) -> Set[Type]:
        """Return an iterable contains all types in this function."""
        seen = set()
        seen.update(self.return_value.contained_types)
        for a in self.arguments:
            seen.update(a.contained_types)
        return seen

    synchrony_spellings = {
        "NW_SYNC": "sync",
        "NW_ASYNC": "async",
        "NW_FLUSH": "flush",
    }

    hidden_annotations = {"api", "location", "name", "return_value", "epilogue", "prologue", "arguments", "type"}

    def __str__(self):
        annotations = ""
        for name, value in self.__dict__.items():
            if name in self.hidden_annotations:
                pass
            elif name == "synchrony" and value in self.synchrony_spellings:
                annotations += f"{_annotation_prefix}{self.synchrony_spellings[value]};\n"
            elif name == "consumes_resources" and value:
                for resource, amount in value.items():
                    annotations += f"{_annotation_prefix}{name}({resource}, {amount});\n"
            elif name == "supported":
                if not value:
                    annotations += f"{_annotation_prefix}unsupported;\n"
            elif not name.startswith("_") and value != default_annotations.get(name):
                if isinstance(value, bool):
                    if value:
                        annotations += f"{_annotation_prefix}{name};\n"
                else:
                    if value:
                        annotations += f"{_annotation_prefix}{name}({value});\n"

        args = ", ".join(str(a) for a in self.real_arguments)
        decl = self.return_value.type.attach_to(f"{self.name}({args})")
        if self.return_value.type.is_void:
            ret_assignment = f"{_annotation_prefix}execute()"
        else:
            ret_assignment = self.return_value.type.attach_to(f"ret = {_annotation_prefix}execute()")
        body = f"""{{\
            {annotations}
            {lines(a.annotations for a in self.arguments)}
            {self.return_value.annotations}
            {lines(str(a) + ";" for a in self.implicit_arguments)}
            {lines(str(a) + ";" for a in self.logue_declarations)}\
            {lines(self.prologue)}\
            {ret_assignment};\
            {lines(self.epilogue)}\
        }}"""
        return f"""{decl}{body}"""


# API

class API(object):
    def __init__(self, name, version, identifier, number, includes: Collection[str],
                 functions, c_types_header_code="", c_utility_code="",
                 metadata_type=None, export_qualifier="", cplusplus=False,
                 **kwds):
        self.name = name
        self.includes = includes
        self.version = version
        self.number = number
        self.identifier = identifier
        self.functions = list(functions)
        self.c_types_header_code = c_types_header_code
        self.c_utility_code = c_utility_code
        self.export_qualifier = export_qualifier
        self.metadata_type = metadata_type
        self.libs = ""
        self.cflags = ""
        self.cxxflags = ""
        self.c_replacement_code = ""
        self.cplusplus = cplusplus

        self.__dict__.update(kwds)

        callback_names = {f.name for f in self.callback_functions}

        for f in functions:
            f.api = self
            parse_requires(all(a.name not in callback_names for a in f.arguments),
                           "A declared callback specification and a function argument may not have the same name.")
        # TODO: Add check to verify that all callback_stubs are actually the names of callback_decls.

    def __str__(self):
        functions = lines(str(f) for f in self.functions)
        includes = self.include_lines
        register_metadata = f"{_annotation_prefix}register_metadata({self.metadata_type.spelling});" if self.metadata_type else ""
        return indent_c(f"""
        {_annotation_prefix}name("{self.name}");
        {_annotation_prefix}version("{self.version}");
        {_annotation_prefix}identifier({self.identifier});
        {_annotation_prefix}number({self.number});
        {_annotation_prefix}export_qualifier({self.export_qualifier});

        {includes}

        {self.c_utility_code}

        {register_metadata}

        {functions}
        """.strip())

    @property
    def include_lines(self):
        return lines(f"#include <{f}>" for f in self.includes)

    @property
    def unsupported_functions(self) -> Iterator[Function]:
        """Generate all the unsupported."""
        return (f for f in self.functions if not f.supported)

    @property
    def supported_functions(self) -> Iterator[Function]:
        """Generate all the unsupported."""
        return (f for f in self.functions if f.supported)

    @property
    def real_functions(self) -> Iterator[Function]:
        """Generate all the application-to-worker API functions."""
        return (f for f in self.functions if not f.callback_decl and f.supported)

    @property
    def callback_functions(self) -> Iterator[Function]:
        """Generate all the worker-to-guest callback functions."""
        return (f for f in self.functions if f.callback_decl and f.supported)

    @property
    def contained_types(self) -> Set[Type]:
        """Return an iterable contains all types in this API. Each type will appear only once (based on Type == Type).
        """
        seen = set()
        for f in self.functions:
            seen.update(f.contained_types)
        return seen

    @property
    def directory_spelling(self) -> str:
        return f"{self.identifier.lower()}_nw"
