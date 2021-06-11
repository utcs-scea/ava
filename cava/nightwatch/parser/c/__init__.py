from typing import Any, List, Dict
import copy

from nightwatch.parser.c.clanginterface import *
from clang.cindex import Diagnostic, Index, LinkageKind, StorageClass
from nightwatch import (
    error, info, warning, location, term, MultipleError
)
from nightwatch.parser.c.rules import (
    Functions,
    Types,
    ConstPointerTypes,
    ConditionalType,
    NonconstPointerTypes,
    PointerTypes,
    NonTransferableTypes,
)
from nightwatch.annotation_set import annotation_set, Conditional
from nightwatch.c_dsl import Expr
from nightwatch.parser.c.util import (
    convert_location,
    extract_annotations,
    extract_attr_annotations,
    get_string_literal,
    strip_unique_suffix,
    strip_nw,
    strip_prefix,
    argument_annotations,
    function_annotations,
    ignored_cursor_kinds,
    resource_directory,
    nightwatch_parser_c_header,
    type_annotations,
    Field,
)
from nightwatch.model import (
    API,
    Argument,
    Function,
    FunctionPointer,
    Location,
    StaticArray,
    Type,
    NIGHTWATCH_PREFIX,
    RET_ARGUMENT_NAME,
    buffer_index_spelling,
)
from nightwatch.parser import parse_assert, parse_requires, parse_expects


consumes_amount_prefix = "consumes_amount_"
allocates_amount_prefix = "allocates_amount_"
deallocates_amount_prefix = "deallocates_amount_"


# pylint: disable=too-many-branches,too-many-statements
def parse(filename: str, include_path: List[str], definitions: List[Any], extra_args: List[Any]) -> API:
    index = Index.create(True)
    includes = [s for p in include_path for s in ["-I", p]]
    definitions = [s for d in definitions for s in ["-D", d]]

    cplusplus = filename.endswith(("cpp", "C", "cc"))

    nightwatch_parser_c_header_fullname = str(resource_directory / nightwatch_parser_c_header)
    llvm_args = (
        includes
        + extra_args
        + clang_flags
        + definitions
        + [
            "-include",
            nightwatch_parser_c_header_fullname,
            f"-D__AVA_PREFIX={NIGHTWATCH_PREFIX}",
            "-x",
            "c++" if cplusplus else "c",
            filename,
        ]
    )
    unit = index.parse(None, args=llvm_args, options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

    errors = []

    severity_table = {
        Diagnostic.Ignored: (None, parse_expects),
        Diagnostic.Note: (info, parse_expects),
        Diagnostic.Warning: (warning, parse_expects),
        Diagnostic.Error: (error, parse_requires),
        Diagnostic.Fatal: (error, parse_requires),
    }

    for d in unit.diagnostics:
        if (
            d.spelling == "incomplete definition of type 'struct __ava_unknown'"
            or d.spelling.startswith("incompatible pointer")
            and d.spelling.endswith("with an expression of type 'struct __ava_unknown *'")
        ):
            continue
        with location("Clang Parser", report_continue=errors):
            kind, func = severity_table[d.severity]
            func(not kind, d.spelling, loc=convert_location(d.location), kind=kind)

    primary_include_files: Dict[str, File] = {}
    primary_include_extents = []
    utility_extents = []
    replacement_extents = []
    type_extents = []

    global_config = {}
    functions: Dict[str, Function] = {}
    include_functions = {}
    replaced_functions = {}
    metadata_type = None

    rules = []
    default_rules = []
    final_rules = []

    def apply_rules(c, annotations, *, name=None):
        if name:
            annotations["name"] = name

        def do(rules):
            for rule in rules:
                rule.apply(c, annotations)

        do(rules)
        if not annotations or (len(annotations) == 1 and "name" in annotations):
            do(default_rules)
        do(final_rules)
        if name:
            del annotations["name"]

    # pylint: disable=too-many-return-statements
    def convert_type(tpe, name, annotations, containing_types):
        parse_requires(
            tpe.get_canonical().spelling not in containing_types or "void" in tpe.get_canonical().spelling,
            "Recursive types don't work.",
        )
        original_containing_types = containing_types
        containing_types = copy.copy(original_containing_types)
        containing_types.add(tpe.get_canonical().spelling)
        parse_assert(tpe.spelling, "Element requires valid and complete type.")
        apply_rules(tpe, annotations, name=name)
        with location(f"in type {term.yellow(tpe.spelling)}"):
            allocates_resources, deallocates_resources = {}, {}
            for annotation_name, annotation_value in annotations.direct().flatten().items():
                if annotation_name.startswith(allocates_amount_prefix):
                    resource = strip_prefix(allocates_amount_prefix, annotation_name)
                    allocates_resources[resource] = annotation_value
                elif annotation_name.startswith(deallocates_amount_prefix):
                    resource = strip_prefix(deallocates_amount_prefix, annotation_name)
                    deallocates_resources[resource] = annotation_value

            parse_expects(
                allocates_resources.keys().isdisjoint(deallocates_resources.keys()),
                "The same argument is allocating and deallocating the same resource.",
            )

            our_annotations = annotations.direct(type_annotations).flatten()
            our_annotations.update(allocates_resources=allocates_resources, deallocates_resources=deallocates_resources)

            if annotations["type_cast"]:
                new_type = annotations["type_cast"]
                # annotations = copy.copy(annotations)
                annotations.pop("type_cast")
                if isinstance(new_type, Conditional):
                    ret = ConditionalType(
                        new_type.predicate,
                        convert_type(new_type.then_branch or tpe, name, annotations, containing_types),
                        convert_type(new_type.else_branch or tpe, name, annotations, containing_types),
                        convert_type(tpe, name, annotations, containing_types),
                    )
                    return ret

                parse_assert(new_type is not None, "ava_type_cast must provide a new type")
                # Attach the original type and then perform conversion using the new type.
                our_annotations["original_type"] = convert_type(
                    tpe, name, annotation_set(), original_containing_types
                )
                tpe = new_type

            if tpe.is_function_pointer():
                pointee = tpe.get_pointee()
                if pointee.kind == TypeKind.FUNCTIONNOPROTO:
                    args = []
                else:
                    args = [convert_type(t, "", annotation_set(), containing_types) for t in pointee.argument_types()]
                return FunctionPointer(
                    tpe.spelling,
                    Type(f"*{name}", **our_annotations),
                    return_type=convert_type(pointee.get_result(), "ret", annotation_set(), containing_types),
                    argument_types=args,
                    **our_annotations,
                )

            if tpe.kind in (TypeKind.FUNCTIONPROTO, TypeKind.FUNCTIONNOPROTO):
                if tpe.kind == TypeKind.FUNCTIONNOPROTO:
                    args = []
                else:
                    args = [convert_type(t, "", annotation_set(), containing_types) for t in tpe.argument_types()]
                return FunctionPointer(
                    tpe.spelling,
                    Type(tpe.spelling, **our_annotations),
                    return_type=convert_type(tpe.get_result(), "ret", annotation_set(), containing_types),
                    argument_types=args,
                    **our_annotations,
                )

            if tpe.is_static_array():
                pointee = tpe.get_pointee()
                pointee_annotations = annotations.subelement("element")
                pointee_name = f"{name}[{buffer_index_spelling}]"
                our_annotations["buffer"] = Expr(tpe.get_array_size())
                return StaticArray(
                    tpe.spelling,
                    pointee=convert_type(pointee, pointee_name, pointee_annotations, containing_types),
                    **our_annotations,
                )

            if tpe.is_pointer():
                pointee = tpe.get_pointee()
                pointee_annotations = annotations.subelement("element")
                pointee_name = f"{name}[{buffer_index_spelling}]"
                if tpe.kind in (TypeKind.VARIABLEARRAY, TypeKind.INCOMPLETEARRAY):
                    sp: str = tpe.spelling
                    sp = sp.replace("[]", "*")
                    return Type(
                        sp,
                        pointee=convert_type(tpe.element_type, pointee_name, pointee_annotations, containing_types),
                        **our_annotations,
                    )
                return Type(
                    tpe.spelling,
                    pointee=convert_type(pointee, pointee_name, pointee_annotations, containing_types),
                    **our_annotations,
                )

            if tpe.get_canonical().kind == TypeKind.RECORD:

                def expand_field(f: Cursor, prefix):
                    f_tpe = f.type
                    decl = f_tpe.get_declaration()
                    if decl.is_anonymous():
                        if decl.kind == CursorKind.UNION_DECL:
                            # FIXME: This assumes the first field is as large or larger than any other field.
                            first_field = sorted(f_tpe.get_fields(), key=lambda f: f.type.get_size())[0]
                            return expand_field(first_field, f"{prefix}.{first_field.displayname}")
                        parse_requires(False, "The only supported anonymous member type is unions.")
                    return [
                        (
                            f.displayname,
                            convert_type(
                                f.type,
                                f"{prefix}.{f.displayname}",
                                annotations.subelement(Field(f.displayname)),
                                containing_types,
                            ),
                        )
                    ]

                field_types = dict(ff for field in tpe.get_canonical().get_fields() for ff in expand_field(field, name))
                return Type(tpe.spelling, fields=field_types, **our_annotations)

            return Type(tpe.spelling, **our_annotations)

    def convert_argument(i, arg, annotations, *, type_=None, is_ret=False):
        name = arg.displayname if not is_ret else RET_ARGUMENT_NAME
        if not name:
            name = "__arg{}".format(i)
        annotations["depends_on"].discard(name)
        apply_rules(arg, annotations, name=name)
        with location(f"argument {term.yellow(name)}", convert_location(arg.location)):
            if not is_ret:
                expressions = list(arg.find_descendants(lambda c: c.kind.is_expression()))
                parse_assert(len(expressions) <= 1, "There must only be one expression child in argument declarations.")
                value = expressions[0].source if expressions else None
            else:
                value = None

            type_ = type_ or arg.type

            return Argument(
                name,
                convert_type(type_, name, annotations, set()),
                value=value,
                location=convert_location(arg.location),
                **annotations.direct(argument_annotations).flatten(),
            )

    def convert_function(cursor, supported=True):
        with location(
            f"at {term.yellow(cursor.displayname)}", convert_location(cursor.location), report_continue=errors
        ):
            # TODO: Capture tokens here and then search them while processing arguments to find commented argument
            #  names.

            body = None
            for c in cursor.get_children():
                if c.kind == CursorKind.COMPOUND_STMT:
                    body = c
                    break

            prologue = []
            epilogue = []
            declarations = []
            implicit_arguments = []
            annotations = annotation_set()
            annotations.update(extract_attr_annotations(cursor))
            if body:
                annotations.update(extract_annotations(body))
                output_list = prologue
                for c in body.get_children():
                    c_annotations = extract_annotations(c)
                    c_attr_annotations = extract_attr_annotations(c)
                    if "implicit_argument" in c_attr_annotations:
                        # FIXME: The [0] should be replaced with code to select the actual correct var decl
                        implicit_arguments.append(c.children[0])
                        continue
                    if len(c_annotations) and list(c_annotations.keys()) != ["depends_on"]:
                        continue
                    found_variables = False
                    if c.kind.is_declaration:
                        for cc in c.find_descendants(lambda cc: cc.kind == CursorKind.VAR_DECL):
                            if not cc.displayname.startswith(NIGHTWATCH_PREFIX) and cc.displayname != "ret":
                                parse_expects(
                                    len(cc.children) == 0,
                                    "Declarations in prologue and epilogue code may not be initialized. "
                                    "(This is currently not checked fully.)",
                                )
                                declarations.append(convert_argument(-2, cc, annotation_set()))
                                found_variables = True
                    if list(c.find_descendants(lambda cc: cc.displayname == "ava_execute")):
                        parse_requires(
                            c.kind != CursorKind.DECL_STMT or c.children[0].displayname == "ret",
                            "The result of ava_execute() must be named 'ret'.",
                        )
                        output_list = epilogue
                    elif not found_variables:
                        src = c.source
                        output_list.append(src + ("" if src.endswith(";") else ";"))
            apply_rules(cursor, annotations, name=cursor.mangled_name)

            args = []
            for i, arg in enumerate(list(cursor.get_arguments()) + implicit_arguments):
                args.append(convert_argument(i, arg, annotations.subelement(arg.displayname)))

            resources = {}
            for annotation_name, annotation_value in annotations.direct().flatten().items():
                if annotation_name.startswith(consumes_amount_prefix):
                    resource = strip_prefix(consumes_amount_prefix, annotation_name)
                    resources[resource] = annotation_value

            return_value = convert_argument(
                -1, cursor, annotations.subelement("return_value"), is_ret=True, type_=cursor.result_type
            )

            if "unsupported" in annotations:
                supported = not bool(annotations["unsupported"])

            disable_native = False
            if "disable_native" in annotations:
                disable_native = bool(annotations["disable_native"])

            return Function(
                cursor.mangled_name,
                return_value,
                args,
                location=convert_location(cursor.location),
                logue_declarations=declarations,
                prologue=prologue,
                epilogue=epilogue,
                consumes_resources=resources,
                supported=supported,
                disable_native=disable_native,
                type=convert_type(cursor.type, cursor.mangled_name, annotation_set(), set()),
                **annotations.direct(function_annotations).flatten(),
            )

    utility_mode = False
    utility_mode_start = None
    replacement_mode = False
    replacement_mode_start = None

    def convert_decl(c: Cursor):
        nonlocal utility_mode, utility_mode_start, replacement_mode, replacement_mode_start, metadata_type
        assert not (replacement_mode and utility_mode)
        if c.kind in ignored_cursor_kinds:
            return

        normal_mode = not replacement_mode and not utility_mode

        # not (c.kind == CursorKind.VAR_DECL and c.displayname.startswith(
        #     NIGHTWATCH_PREFIX)) and (utility_mode or replacement_mode):
        included_extent = True
        if (
            normal_mode
            and c.kind == CursorKind.FUNCTION_DECL
            and c.location.file.name == filename
            and c.spelling == "ava_metadata"
        ):
            metadata_type = convert_type(c.result_type.get_pointee(), "ava_metadata", annotation_set(), set())
        elif (
            normal_mode
            and c.kind == CursorKind.FUNCTION_DECL
            and c.displayname.startswith(NIGHTWATCH_PREFIX + "category_")
        ):
            name = strip_unique_suffix(strip_prefix(NIGHTWATCH_PREFIX + "category_", c.displayname))
            annotations = extract_annotations(c)
            attr_annotations = extract_attr_annotations(c)
            rule_list = default_rules if "default" in attr_annotations else rules
            annotations.pop("default", None)
            if name == "type":
                rule_list.append(Types(c.result_type.get_pointee(), annotations))
            elif name == "functions":
                rule_list.append(Functions(annotations))
            elif name == "pointer_types":
                rule_list.append(PointerTypes(annotations))
            elif name == "const_pointer_types":
                rule_list.append(ConstPointerTypes(annotations))
            elif name == "nonconst_pointer_types":
                rule_list.append(NonconstPointerTypes(annotations))
            elif name == "non_transferable_types":
                rule_list.append(NonTransferableTypes(annotations))
        elif normal_mode and c.kind == CursorKind.VAR_DECL and c.storage_class == StorageClass.STATIC:
            # This is a utility function for the API forwarding code.
            parse_expects(
                c.linkage == LinkageKind.INTERNAL,
                f"at {term.yellow(c.displayname)}",
                "API utility functions should be static (or similar) since they are included in header files.",
                loc=convert_location(c.location),
            )
            utility_extents.append((c.extent.start.line, c.extent.end.line))
        elif c.kind == CursorKind.VAR_DECL and c.displayname.startswith(NIGHTWATCH_PREFIX):
            name = strip_unique_suffix(strip_nw(c.displayname))
            if name == "begin_utility":
                parse_requires(
                    not utility_mode, "ava_begin_utility can only be used outside utility mode to enter that mode."
                )
                utility_mode = True
                utility_mode_start = c.extent.start.line
            elif name == "end_utility":
                parse_requires(utility_mode, "ava_end_utility can only be used inside utility mode to exit that mode.")
                utility_mode = False
                parse_assert(utility_mode_start is not None, "Should be unreachable.")
                utility_extents.append((utility_mode_start, c.extent.end.line))
            elif name == "begin_replacement":
                parse_requires(
                    not replacement_mode,
                    "ava_begin_replacement can only be used outside replacement mode to enter that mode.",
                )
                replacement_mode = True
                replacement_mode_start = c.extent.start.line
            elif name == "end_replacement":
                parse_requires(
                    replacement_mode, "ava_end_replacement can only be used inside replacement mode to exit that mode."
                )
                replacement_mode = False
                parse_assert(replacement_mode_start is not None, "Should be unreachable.")
                replacement_extents.append((replacement_mode_start, c.extent.end.line))
            else:
                global_config[name] = get_string_literal(c)
        elif (
            normal_mode
            and c.kind == CursorKind.VAR_DECL
            and c.type.spelling.endswith("_resource")
            and c.type.spelling.startswith("ava_")
        ):
            # TODO: Use the resource declarations to check resource usage.
            pass
        elif c.kind == CursorKind.FUNCTION_DECL and c.location.file.name == filename:
            if normal_mode and c.is_definition() and c.storage_class == StorageClass.STATIC:
                # This is a utility function for the API forwarding code.
                parse_expects(
                    c.linkage == LinkageKind.INTERNAL,
                    f"at {term.yellow(c.displayname)}",
                    "API utility functions should be static (or similar) since they are included in header files.",
                    loc=convert_location(c.location),
                )
                utility_extents.append((c.extent.start.line, c.extent.end.line))
            elif normal_mode:
                # This is an API function.
                f = convert_function(c)
                if f:
                    functions[c.mangled_name] = f
            elif replacement_mode:
                # Remove the function from the list because it is replaced
                replaced_functions[c.mangled_name] = c
        elif (
            normal_mode
            and c.kind == CursorKind.FUNCTION_DECL
            and c.location.file.name in [f.name for f in primary_include_files.values()]
        ):
            included_extent = False
            f = convert_function(c, supported=False)
            if f:
                include_functions[c.mangled_name] = f
        elif (
            normal_mode
            and c.kind == CursorKind.INCLUSION_DIRECTIVE
            and not c.displayname.endswith(nightwatch_parser_c_header)
            and c.location.file.name == filename
        ):
            try:
                primary_include_files[c.displayname] = c.get_included_file()
            except AssertionError as e:
                parse_assert(not e, str(e), loc=convert_location(c.location))
        # elif normal_mode and c.kind == CursorKind.INCLUSION_DIRECTIVE and c.tokens[-1].spelling == '"' \
        #         and not c.displayname.endswith(nightwatch_parser_c_header):
        #     parse_assert(False, "Including AvA specifications in other specifications is not yet supported.")
        elif (
            normal_mode
            and c.kind in (CursorKind.MACRO_DEFINITION, CursorKind.STRUCT_DECL, CursorKind.TYPEDEF_DECL)
            and c.location.file
            and c.location.file.name == filename
        ):
            # This is a utility macro for the API forwarding code.
            type_extents.append((c.extent.start.line, c.extent.end.line))
        elif (
            # pylint: disable=too-many-boolean-expressions
            (normal_mode or replacement_mode)
            and c.kind in (CursorKind.UNEXPOSED_DECL,)
            and len(c.tokens)
            and c.tokens[0].spelling == "extern"
            and c.location.file in primary_include_files.values()
        ):
            for cc in c.get_children():
                convert_decl(cc)
            return  # Skip the extents processing below
        elif normal_mode:
            # Default case for normal mode.
            is_semicolon = len(c.tokens) == 1 and c.tokens[0].spelling == ";"
            if c.location.file and not is_semicolon:
                parse_expects(
                    c.location.file.name != filename,
                    f"Ignoring unsupported: {c.kind} {c.spelling}",
                    loc=convert_location(c.location),
                )
            # if len(c.tokens) >= 1 and c.tokens[0].spelling == "extern" and c.kind == CursorKind.UNEXPOSED_DECL:
            #     print(c.kind, c.tokens[0].spelling)
        else:
            # Default case for non-normal modes
            return  # Skip the extents processing below

        if c.location.file in primary_include_files.values():
            primary_include_extents.append((c.location.file, c.extent.start.line, c.extent.end.line, included_extent))

    for c in unit.cursor.get_children():
        convert_decl(c)

    parse_expects(primary_include_files, "Expected at least one API include file.")

    extra_functions = {}

    if errors:
        raise MultipleError(*errors)

    for name, function in functions.items():
        if name in include_functions:
            del include_functions[name]
        elif not function.callback_decl:
            extra_functions[name] = function

    for name, cursor in replaced_functions.items():
        if name in include_functions:
            del include_functions[name]
        else:
            parse_requires(
                name not in functions,
                "Replacing forwarded functions is not allowed.",
                loc=convert_location(cursor.location),
            )

    if extra_functions:
        function_str = ", ".join(str(f.name) for f in extra_functions.values())
        parse_expects(
            False,
            f"""
Functions appear in {filename}, but are not in {", ".join(primary_include_files.keys())}:
{function_str}""".strip(),
            loc=Location(filename, None, None, None),
        )

    # We use binary mode because clang operates in bytes not characters.
    # TODO: If the source files have "\r\n" and clang uses text mode then this will cause incorrect removals.
    # TODO: There could be functions in the header which are not processed with the current configuration. That will
    #  mess things up.
    c_types_header_code = bytearray()
    for name, file in primary_include_files.items():
        with open(file.name, "rb") as fi:
            # content = fi.read()
            # primary_include_extents.sort(key=lambda r: r(0).start.offset)
            def find_modes(i):
                modes = set()
                for in_name, start, end, mode in primary_include_extents:
                    # pylint: disable=cell-var-from-loop
                    if in_name == file and start <= i <= end:
                        modes.add(mode)
                return modes

            error_reported = False
            i = None
            for i, line in enumerate(fi):
                modes = find_modes(i + 1)
                # print(i, modes, line)
                keep_line = True in modes or not modes
                error_line = keep_line and False in modes
                parse_expects(
                    not error_line or error_reported,
                    "Line both needed and excluded. Incorrect types header may be generated.",
                    loc=Location(file.name, i, None, None),
                )
                error_reported = error_reported or error_line
                if keep_line:
                    c_types_header_code.extend(line)
                else:
                    c_types_header_code.extend(
                        b"/* NWR: " + line.replace(b"/*", b"@*").replace(b"*/", b"*@").rstrip() + b" */\n"
                    )

    def load_extents(extents):
        with open(filename, "rb") as fi:

            def utility_line(i):
                for start, end in extents:
                    if start <= i <= end:
                        return True
                return False

            c_code = bytearray()
            last_emitted_line = None
            for i, line in enumerate(fi):
                if utility_line(i + 1):
                    if last_emitted_line != i - 1:
                        c_code.extend("""#line {} "{}"\n""".format(i + 1, filename).encode("utf-8"))
                    c_code.extend(line)
                    last_emitted_line = i
        return bytes(c_code).decode("utf-8")

    c_utility_code = load_extents(utility_extents)
    c_replacement_code = load_extents(replacement_extents)
    c_type_code = load_extents(type_extents)

    return API(
        functions=list(functions.values()) + list(include_functions.values()),
        includes=list(primary_include_files.keys()),
        c_types_header_code=bytes(c_types_header_code).decode("utf-8"),
        c_type_code=c_type_code,
        c_utility_code=c_utility_code,
        c_replacement_code=c_replacement_code,
        metadata_type=metadata_type,
        missing_functions=list(include_functions.values()),
        cplusplus=cplusplus,
        **global_config,
    )
