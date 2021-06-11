from nightwatch import location, term
from nightwatch.c_dsl import Expr
from nightwatch.generator.c.buffer_handling import get_transfer_buffer_expr
from nightwatch.generator.c.util import for_all_elements
from nightwatch.generator.common import unpack_struct, snl
from nightwatch.model import Function, Type, Argument, API, lines


def command_print_implementation(f: Function):
    with location(f"at {term.yellow(str(f.name))}", f.location):

        def printf(fmt, *values):
            return f"""fprintf(file, "{fmt}", {",".join(values)});"""

        def print_value_deep(values, cast_type: Type, type_: Type, depth, no_depends, argument, **other):
            (value,) = values
            if type_.is_void:
                return ""
            buffer_pred = Expr(type_.transfer).equals("NW_BUFFER") & Expr(value).not_equals("NULL")

            def address():
                if not hasattr(type_, "pointee"):
                    return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
                tmp_name = f"__tmp_{argument.name}_{depth}"
                inner_values = (tmp_name,)
                data_code = buffer_pred.if_then_else(
                    f"""
                    fprintf(file, " = {{");
                    {type_.nonconst.attach_to(tmp_name)};
                    {tmp_name} = ({cast_type})({get_transfer_buffer_expr(value, type_)});
                    {for_all_elements(
                        inner_values,
                        cast_type,
                        type_,
                        precomputed_size=Expr(1),
                        depth=depth,
                        argument=argument,
                        no_depends=no_depends,
                        **other
                    )}
                    fprintf(file, ",...}}");
                    """
                )
                return f"""
                {printf("ptr 0x%012lx", f"(long int){value}")}
                {data_code}
                """

            def handle():
                return printf("handle %#lx", f"(long int){value}")

            def opaque():
                st = str(type_)
                if "*" in st:
                    return printf("%#lx", f"(long int){value}")
                if "int" in st:
                    return printf("%ld", f"(long int){value}")
                if "float" in st or "double" in st:
                    return printf("%Lf", f"(long double){value}")
                # Fall back on pointer representation
                return printf("%#lx", f"(long int){value}")

            return Expr(bool(type_.fields or argument.depends_on and no_depends)).if_then_else(
                "",  # Using only else branch
                Expr(type_.transfer)
                .equals("NW_BUFFER")
                .if_then_else(
                    address,
                    Expr(type_.transfer)
                    .equals("NW_ZEROCOPY_BUFFER")
                    .if_then_else(
                        address,
                        Expr(type_.transfer)
                        .equals("NW_OPAQUE")
                        .if_then_else(opaque, Expr(type_.transfer).equals("NW_HANDLE").if_then_else(handle)),
                    ),
                ),
            )

        def print_value(argument: Argument, value, no_depends):
            conv = print_value_deep(
                (value,),
                argument.type.nonconst,
                argument.type,
                depth=0,
                name=argument.name,
                argument=argument,
                no_depends=no_depends,
                kernel=print_value_deep,
                self_index=0,
            )
            return (printf("%s=", f'"{argument.name}"') if not argument.ret else "") + str(conv)

        print_comma = """ fprintf(file, ", ");\n"""
        return f"""
        case {f.call_id_spelling}: {{ \
            ava_is_in = 1; ava_is_out = 0;
            struct {f.call_spelling}* __call = (struct {f.call_spelling}*)__cmd;
            assert(__call->base.api_id == {f.api.number_spelling});
            assert(__call->base.command_size == sizeof(struct {f.call_spelling}) && "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            {unpack_struct("__call", f.arguments, "->", get_transfer_buffer_expr)}
            {printf("<%03ld> <thread=%012lx> %s(", "(long int)__call->__call_id", "(unsigned long int)__call->base.thread_id", f'"{f.name}"')}
            {print_comma.join(str(print_value(a, f"__call->{a.name}", False)) for a in f.arguments if a.input or not a.type.contains_buffer)}
            fprintf(file, "){snl}");
            break;
        }}
        case {f.ret_id_spelling}: {{ \
            ava_is_in = 0; ava_is_out = 1;
            struct {f.ret_spelling}* __ret = (struct {f.ret_spelling}*)__cmd;
            assert(__ret->base.api_id == {f.api.number_spelling});
            assert(__ret->base.command_size == sizeof(struct {f.ret_spelling}) && "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            {unpack_struct("__ret",
                           ([] if f.return_value.type.is_void else [f.return_value]) +
                           [a for a in f.arguments if a.output and a.type.contains_buffer and not bool(a.depends_on)],
                           "->", get_transfer_buffer_expr)}
            {printf("<%03ld> <thread=%012lx> %s(", "(long int)__ret->__call_id", "(unsigned long int)__ret->base.thread_id", f'"{f.name}"')}
            {print_comma.join(str(print_value(a, f"__ret->{a.name}", True)) for a in f.arguments if a.output and a.type.contains_buffer)}
            fprintf(file, ") -> ");
            {print_value(f.return_value, f"__ret->{f.return_value.name}", True)}
            fprintf(file, "{snl}");
            break;
        }}
        """.strip()


def print_command_function(api: API) -> str:
    function_name = f"__print_command_{api.identifier.lower()}"
    return f"""
    void {function_name}(FILE* file, const struct command_channel* __chan, const struct command_base* __cmd) {{
        int ava_is_in, ava_is_out;
        switch (__cmd->command_id) {{
        {lines(command_print_implementation(f) for f in api.supported_functions)}
        default:
            abort_with_reason("Received unsupported command");
        }} // switch
    }}
    """.strip()
