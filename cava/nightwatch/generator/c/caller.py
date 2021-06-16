from nightwatch import location, term
from nightwatch.c_dsl import ExprOrStr, Expr
from nightwatch.generator import generate_requires, generate_expects
from nightwatch.generator.c.buffer_handling import (
    get_buffer,
    get_transfer_buffer_expr,
    attach_buffer,
    get_buffer_expr,
    deallocate_managed_for_argument,
    size_to_bytes,
    allocate_tmp_buffer,
    DECLARE_BUFFER_SIZE_EXPR,
)
from nightwatch.generator.c.util import compute_buffer_size, for_all_elements, AllocList
from nightwatch.generator.common import comment_block, unpack_struct
from nightwatch.model import Argument, Type, ConditionalType, Function, lines
from nightwatch.generator.c.instrumentation import timing_code_guest


def copy_result_for_argument(arg: Argument, dest: str, src: str) -> ExprOrStr:
    """
    Copy arg from the src struct into dest struct.
    :param arg: The argument to copy.
    :param dest: The destination call record struct for the call.
    :param src: The source RET struct from the call.
    :return: A series C statements.
    """
    reported_missing_lifetime = False

    def convert_result_value(values, cast_type: Type, type_: Type, depth, original_type=None, **other):
        if isinstance(type_, ConditionalType):
            return Expr(type_.predicate).if_then_else(
                convert_result_value(
                    values, type_.then_type.nonconst, type_.then_type, depth, original_type=type_.original_type, **other
                ),
                convert_result_value(
                    values, type_.else_type.nonconst, type_.else_type, depth, original_type=type_.original_type, **other
                ),
            )

        param_value, local_value = values

        src_name = f"__src_{arg.name}_{depth}"

        def get_buffer_code():
            nonlocal reported_missing_lifetime
            if not reported_missing_lifetime:
                if ((arg.ret or arg.output and depth > 0) and type_.buffer) and type_.lifetime == "AVA_CALL":
                    reported_missing_lifetime = True
                    generate_expects(
                        False,
                        "Returned buffers with call lifetime are almost always incorrect. "
                        "(You may want to set a lifetime.)",
                    )
            return Expr(
                f"""
                {DECLARE_BUFFER_SIZE_EXPR}
                {type_.attach_to(src_name)};
                {src_name} = ({type_.spelling})({get_transfer_buffer_expr(local_value, type_, not_null=True)});
                """
            ).then(
                Expr(type_.lifetime)
                .not_equals("AVA_CALL")
                .if_then_else(
                    f"""{get_buffer(
                            param_value,
                            cast_type,
                            local_value,
                            type_,
                            original_type=original_type,
                            not_null=True,
                            declare_buffer_size=False
                        )}""",
                    f"""__buffer_size = {compute_buffer_size(type_, original_type)};""",
                )
                .then(Expr(arg.output).if_then_else(f"AVA_DEBUG_ASSERT({param_value} != NULL);"))
            )

        def simple_buffer_case():
            if not hasattr(type_, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            copy_code = Expr(arg.output).if_then_else(
                f"""memcpy({param_value}, {src_name}, {size_to_bytes("__buffer_size", type_)});"""
            )
            if copy_code:
                return (
                    Expr(local_value)
                    .not_equals("NULL")
                    .if_then_else(
                        f"""
                    {get_buffer_code()}
                    {copy_code}
                    """.strip()
                    )
                )
            return ""

        def buffer_case():
            if not hasattr(type_, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            if not arg.output:
                return simple_buffer_case()

            inner_values = (param_value, src_name)
            loop = for_all_elements(
                inner_values,
                cast_type,
                type_,
                depth=depth,
                precomputed_size="__buffer_size",
                original_type=original_type,
                **other,
            )
            if loop:
                return (
                    Expr(local_value)
                    .not_equals("NULL")
                    .if_then_else(
                        f"""
                    {get_buffer_code()}
                    {loop}
                    """
                    )
                )
            return ""

        def default_case():
            dealloc_code = (Expr(type_.transfer).equals("NW_HANDLE") & type_.deallocates).if_then_else(
                f"""
                ava_coupled_free(&__ava_endpoint, {local_value});
                """.strip()
            )
            return dealloc_code.then((Expr(arg.output) | arg.ret).if_then_else(f"{param_value} = {local_value};"))

        if type_.fields:
            return for_all_elements(values, cast_type, type_, depth=depth, original_type=original_type, **other)
        return (
            type_.is_simple_buffer(allow_handle=False)
            .if_then_else(
                simple_buffer_case,
                Expr(type_.transfer)
                .equals("NW_BUFFER")
                .if_then_else(
                    buffer_case, Expr(type_.transfer).one_of({"NW_OPAQUE", "NW_HANDLE"}).if_then_else(default_case)
                ),
            )
            .scope()
        )

    with location(f"at {term.yellow(str(arg.name))}", arg.location):
        conv = convert_result_value(
            (f"{dest}->{arg.param_spelling}", f"{src}->{arg.name}"),
            arg.type.nonconst,
            arg.type,
            depth=0,
            name=arg.name,
            kernel=convert_result_value,
            self_index=0,
        )
        return comment_block(f"Output: {arg}", conv)


def compute_argument_value(arg: Argument):
    if arg.implicit_argument:
        return f"""
        {arg.type.nonconst.attach_to(arg.name)};
        {arg.name} = {arg.value};
        """.strip()
    return ""


def attach_for_argument(arg: Argument, dest: str):
    """
    Copy arg into dest attaching buffers as needed.
    :param arg: The argument to copy.
    :param dest: The destination CALL struct.
    :return: A series of C statements.
    """
    alloc_list = AllocList(arg.function)

    def copy_for_value(values, cmd_value_type: Type, type_: Type, depth, argument, original_type=None, **other):
        if isinstance(type_, ConditionalType):
            return Expr(type_.predicate).if_then_else(
                copy_for_value(
                    values,
                    type_.then_type.nonconst,
                    type_.then_type,
                    depth,
                    argument,
                    original_type=type_.original_type,
                    **other,
                ),
                copy_for_value(
                    values,
                    type_.else_type.nonconst,
                    type_.else_type,
                    depth,
                    argument,
                    original_type=type_.original_type,
                    **other,
                ),
            )

        arg_value, cmd_value = values

        def attach_data(data):
            return attach_buffer(
                cmd_value,
                cmd_value_type,
                arg_value,
                data,
                type_,
                arg.input,
                cmd=dest,
                original_type=original_type,
                expect_reply=True,
            )

        def simple_buffer_case():
            if not hasattr(type_, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            return (Expr(arg_value).not_equals("NULL") & (Expr(type_.buffer) > 0)).if_then_else(
                attach_data(arg_value), f"{cmd_value} = NULL;"
            )

        def buffer_case():
            if not hasattr(type_, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            if not arg.input:
                return simple_buffer_case()

            tmp_name = f"__tmp_{arg.name}_{depth}"
            size_name = f"__size_{arg.name}_{depth}"
            loop = for_all_elements(
                (arg_value, tmp_name),
                cmd_value_type,
                type_,
                depth=depth,
                argument=argument,
                precomputed_size=size_name,
                original_type=original_type,
                **other,
            )
            return (Expr(arg_value).not_equals("NULL") & (Expr(type_.buffer) > 0)).if_then_else(
                f"""
                    {allocate_tmp_buffer(
                        tmp_name,
                        size_name,
                        type_,
                        alloc_list=alloc_list,
                        original_type=original_type
                    )}
                    {loop}
                    {attach_data(tmp_name)}
                """,
                f"{cmd_value} = NULL;",
            )

        def default_case():
            return Expr(not type_.is_void).if_then_else(
                f"{cmd_value} = ({cmd_value_type}){arg_value};",
                """abort_with_reason("Reached code to handle void value.");""",
            )

        if type_.fields:
            return for_all_elements(
                values, cmd_value_type, type_, depth=depth, argument=argument, original_type=original_type, **other
            )
        return (
            type_.is_simple_buffer(allow_handle=True)
            .if_then_else(
                simple_buffer_case, Expr(type_.transfer).equals("NW_BUFFER").if_then_else(buffer_case, default_case)
            )
            .scope()
        )

    with location(f"at {term.yellow(str(arg.name))}", arg.location):
        userdata_code = ""
        if arg.userdata and not arg.function.callback_decl:
            try:
                (callback_arg,) = [a for a in arg.function.arguments if a.type.transfer == "NW_CALLBACK"]
            except ValueError:
                generate_requires(
                    False,
                    "If ava_userdata is applied to an argument exactly one other argument "
                    "must be annotated with ava_callback.",
                )
            generate_requires(
                [arg] == [a for a in arg.function.arguments if a.userdata],
                "Only one argument on a given function can be annotated with ava_userdata.",
            )
            userdata_code = f"""
            if ({callback_arg.param_spelling} != NULL) {{
                // TODO:MEMORYLEAK: This leaks 2*sizeof(void*) whenever a callback is transported. Should be fixable
                //  with "coupled buffer" framework.
                struct ava_callback_user_data *__callback_data = malloc(sizeof(struct ava_callback_user_data));
                __callback_data->userdata = {arg.param_spelling};
                __callback_data->function_pointer = (void*){callback_arg.param_spelling};
                {arg.param_spelling} = __callback_data;
            }}
            """
        return comment_block(
            f"Input: {arg}",
            Expr(userdata_code).then(
                copy_for_value(
                    (arg.param_spelling, f"{dest}->{arg.param_spelling}"),
                    arg.type.nonconst,
                    arg.type,
                    depth=0,
                    argument=arg,
                    name=arg.name,
                    kernel=copy_for_value,
                    only_complex_buffers=False,
                    self_index=0,
                )
            ),
        )


def return_command_implementation(f: Function):
    with location(f"at {term.yellow(str(f.name))}", f.location):
        generate_requires(
            not f.return_value.type.buffer or f.return_value.type.lifetime != Expr("AVA_CALL"),
            "Returned buffers must have a lifetime other than `call' "
            "(i.e., must be annotated with `ava_lifetime_static', `ava_lifetime_coupled', or `ava_lifetime_manual').",
        )
        return f"""
        case {f.ret_id_spelling}: {{\
            {timing_code_guest("before_unmarshal", str(f.name), f.generate_timing_code)}
            ava_is_in = 0; ava_is_out = 1;
            struct {f.ret_spelling}* __ret = (struct {f.ret_spelling}*)__cmd;
            assert(__ret->base.api_id == {f.api.number_spelling});
            assert(__ret->base.command_size == sizeof(struct {f.ret_spelling}) && "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            struct {f.call_record_spelling}* __local = (struct {f.call_record_spelling}*)ava_remove_call(&__ava_endpoint, __ret->__call_id);

            {{
                {unpack_struct("__local", f.arguments, "->")} \
                {unpack_struct("__local", f.logue_declarations, "->")} \
                {unpack_struct("__ret", [f.return_value], "->", convert=get_buffer_expr)
                    if not f.return_value.type.is_void else ""} \
                {lines(copy_result_for_argument(a, "__local", "__ret")
                       for a in f.arguments if a.type.contains_buffer)}
                {copy_result_for_argument(f.return_value, "__local", "__ret") if not f.return_value.type.is_void else ""}\
                {lines(f.epilogue)}
                {lines(deallocate_managed_for_argument(a, "__local")
                       for a in f.arguments)}
            }}

            {timing_code_guest("after_unmarshal", str(f.name), f.generate_timing_code)}
            __local->__call_complete = 1;
            if(__local->__handler_deallocate) {{
                free(__local);
            }}
            break;
        }}
        """.strip()
