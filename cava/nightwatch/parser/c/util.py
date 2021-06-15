from pathlib import Path

import ast
import re
from collections import namedtuple
from typing import Optional, Tuple, Union, Set

# pylint: disable=unused-import
import nightwatch.parser.c.reload_libclang
from clang.cindex import Cursor, CursorKind, SourceLocation
from nightwatch.parser.c.clanginterface import _CursorExtension
from nightwatch.annotation_set import AnnotationSet, annotation_set
from nightwatch.c_dsl import Expr
from nightwatch.model import Location, NIGHTWATCH_PREFIX
from nightwatch.parser import location, parse_assert


def strip_prefix(prefix: str, s: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


UNIQUE_SUFFIX = re.compile(r"_+\d+(\(\))?$")


def strip_unique_suffix(s: str) -> str:
    r = UNIQUE_SUFFIX.sub("", s)
    return r


def strip_nw(s: str) -> str:
    return strip_prefix(NIGHTWATCH_PREFIX, s)


def maybe_parse(s):
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return s


NW_ANNOTATION_RE = re.compile(r"(?P<name>\w+)(\((?P<arguments>.*)\))?")
NW_ANNOTATION_SPLIT_RE = re.compile(r"\s*,\s*")


def parse_annotation(s: str) -> Tuple[str, bool]:
    match = NW_ANNOTATION_RE.fullmatch(s)
    name = match.group("name")
    if match.group("arguments") is not None:
        value = match.group("arguments")
    else:
        value = True
    return name, value


attr_annotation_relevant_kinds = frozenset((CursorKind.VAR_DECL, CursorKind.FUNCTION_DECL))


def extract_attr_annotations(c: Cursor) -> AnnotationSet:
    d = annotation_set()
    for cc in c.find_descendants(lambda c: c.kind in attr_annotation_relevant_kinds):
        annotation_nodes = [
            strip_nw(a.spelling)
            for a in cc.get_children()
            if a.kind == CursorKind.ANNOTATE_ATTR and a.spelling.startswith(NIGHTWATCH_PREFIX)
        ]
        for s in annotation_nodes:
            name, value = parse_annotation(s)
            d[name] = value
    return d


def get_string_literal(c: Cursor) -> Optional[str]:
    if c.kind == CursorKind.STRING_LITERAL:
        # TODO: This parses python literals which are not the same as C. Ideally this would use clang to parse C
        #  literals to bytes.
        return ast.literal_eval(c.spelling)

    for ch in c.get_children():
        return get_string_literal(ch)


def convert_location(loc: SourceLocation) -> Location:
    if hasattr(loc, "location"):
        return convert_location(loc.location)
    if hasattr(loc, "file") and hasattr(loc, "line"):
        return Location(loc.file and loc.file.name, loc.line, loc.column, loc.offset)
    return None


# Annotation lists

resource_directory = Path(__file__).parent
nightwatch_parser_c_header = "nightwatch.h"

function_annotations = {
    "synchrony",
    "ignore",
    "callback_decl",
    "object_record",
    "generate_timing_code",
    "generate_stats_code",
}
type_annotations = {
    "transfer",
    "success",
    "name",
    "element",
    "deallocates",
    "allocates",
    "buffer",
    "object_explicit_state_extract",
    "object_explicit_state_replace",
    "buffer_allocator",
    "buffer_deallocator",
    "object_record",
    "object_depends_on",
    "callback_stub_function",
    "lifetime",
    "lifetime_coupled",
}
argument_annotations = {"depends_on", "value", "implicit_argument", "input", "output", "no_copy", "userdata"}

ignored_cursor_kinds = frozenset([CursorKind.MACRO_INSTANTIATION])

LIBCLANG_INFO = (
    "(INFO: This can be caused by using an unpatched libclang. "
    "You must use the clang code in the nightwatch-combined repo.)"
)


def _as_bool(s: str) -> bool:
    if not s:
        raise ValueError("Empty boolean is not allowed. " + LIBCLANG_INFO)
    return bool(int(s))


def _as_string_set(s) -> Set:
    return {s for s in NW_ANNOTATION_SPLIT_RE.split(s and ast.literal_eval(s)) if s}


def _as_cexpr_singleton_set(s):
    return set([Expr(s)])


annotation_parsers = dict(
    depends_on=_as_string_set,
    object_depends_on=_as_cexpr_singleton_set,
    object_record=_as_bool,
    object_explicit_state_extract=Expr,
    object_explicit_state_replace=Expr,
    buffer_allocator=Expr,
    buffer_deallocator=Expr,
    input=_as_bool,
    output=_as_bool,
    no_copy=_as_bool,
    allocates=_as_bool,
    deallocates=_as_bool,
    buffer=Expr,
    transfer=Expr,
    userdata=_as_bool,
    callback_stub_function=Expr,
    lifetime=Expr,
    lifetime_coupled=Expr,
    generate_timing_code=_as_bool,
    generate_stats_code=_as_bool,
)

annotation_relevant_kinds = frozenset((CursorKind.VAR_DECL, CursorKind.IF_STMT))


def annotation_parser(name: str):
    if (
        name.startswith("consumes_amount_")
        or name.startswith("allocates_amount_")
        or name.startswith("deallocates_amount_")
    ):
        return Expr
    return annotation_parsers.get(name, lambda x: x)


def extract_predicate(cursor: Cursor) -> Union[Tuple[str, str], str]:
    if cursor.kind == CursorKind.BINARY_OPERATOR and cursor.children[0].spelling.startswith(NIGHTWATCH_PREFIX):
        name = strip_prefix(NIGHTWATCH_PREFIX, cursor.children[0].spelling)
        return name, cursor.children[1].untokenized
    if cursor.spelling.startswith(NIGHTWATCH_PREFIX):
        name = strip_prefix(NIGHTWATCH_PREFIX, cursor.spelling)
        return name, ""
    return cursor.unparsed


class Field(namedtuple("Field", ["name"])):
    pass


def extract_annotations(cursor: Cursor) -> AnnotationSet:
    ret = annotation_set()
    for c in cursor.find_descendants(lambda c: c.kind in annotation_relevant_kinds):
        with location(convert_location(c.location)):
            ret["depends_on"] = set(c.referenced_parameters) | ret.get("depends_on", set())
            if c.kind == CursorKind.VAR_DECL:
                ret.update(extract_attr_annotations(c).pushdown(c.displayname))
            if c.kind == CursorKind.VAR_DECL and c.displayname.startswith(NIGHTWATCH_PREFIX):
                name = strip_prefix(NIGHTWATCH_PREFIX, c.displayname)
                if len(c.children) > 0 and c.children[-1].kind.is_expression():
                    expr = c.children[-1]
                    ret[name] = annotation_parser(name)(expr.unparsed)
                    # ret["depends_on"] = set(expr.referenced_parameters) | ret.get("depends_on", set())
                else:
                    parse_assert(
                        name == "type_cast", f"Missing value for annotation variable declaration: {c.unparsed}"
                    )
                    ret[name] = c.type
            elif c.kind == CursorKind.IF_STMT:
                pred = extract_predicate(c.children[0])
                then_branch = extract_annotations(c.children[1])
                else_branch = extract_annotations(c.children[2]) if len(c.children) > 2 else annotation_set()
                if isinstance(pred, tuple) and pred[0] == "argument_block":
                    assert not else_branch
                    ret.update(then_branch.pushdown(pred[1]))
                elif isinstance(pred, tuple) and pred[0] == "return_value_block":
                    assert not else_branch
                    ret.update(then_branch.pushdown("return_value"))
                elif isinstance(pred, tuple) and pred[0] == "element_block":
                    assert not else_branch
                    ret.update(then_branch.pushdown("element"))
                elif isinstance(pred, tuple) and pred[0].startswith("field_block_"):
                    name = strip_prefix("field_block_", pred[0])
                    ret.update(then_branch.pushdown(Field(name)))
                else:
                    ret.update(then_branch.if_else(pred, else_branch))
    return ret
