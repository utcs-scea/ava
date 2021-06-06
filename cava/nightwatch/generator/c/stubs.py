from nightwatch import location, term
from nightwatch.c_dsl import Expr, ExprOrStr
from nightwatch.generator import generate_requires
from nightwatch.generator.c.buffer_handling import compute_total_size
from nightwatch.generator.c.caller import compute_argument_value, attach_for_argument
from nightwatch.generator.c.instrumentation import timing_code_guest, report_alloc_resources, report_consume_resources
from nightwatch.generator.c.util import *
from nightwatch.generator.common import *
from nightwatch.model import *
from typing import Union


def function_implementation(f: Function) -> Union[str, Expr]:
    """
    Generate a stub function which sends the appropriate CALL command over the
    channel.
    :param f: The function to generate a stub for.
    :return: A C function definition (as a string or Expr)
    """
    with location(f"at {term.yellow(str(f.name))}", f.location):
        if f.return_value.type.buffer:
            forge_success = f"#error Async returned buffers are not implemented."
        elif f.return_value.type.is_void:
            forge_success = "return;"
        elif f.return_value.type.success is not None:
            forge_success = f"return {f.return_value.type.success};"
        else:
            forge_success = """abort_with_reason("Cannot forge success without a success value for the type.");"""

        if f.return_value.type.is_void:
            return_statement = f"""
                free(__call_record);
                return;
            """.strip()
        else:
            return_statement = f"""
                {f.return_value.declaration};
                {f.return_value.name} = __call_record->{f.return_value.name};
                free(__call_record);
                return {f.return_value.name};
            """.strip()

        is_async = ~Expr(f.synchrony).equals("NW_SYNC")

        alloc_list = AllocList(f)

        send_code = f"""
            command_channel_send_command(__chan, (struct command_base*)__cmd);
        """.strip()

        if f.api.send_code:
            import_code = f.api.send_code.encode("ascii", "ignore").decode("unicode_escape")[1:-1]
            ldict = locals()
            exec(import_code, globals(), ldict)
            send_code = ldict["send_code"]

        return_code = is_async.if_then_else(
            forge_success,
            f"""
                shadow_thread_handle_command_until(nw_shadow_thread_pool, __call_record->__call_complete);
                {return_statement}
            """.strip(),
        )

        return f"""
        EXPORTED {(f.api.export_qualifier + " ") if f.api.export_qualifier else ""}{f.return_value.type.spelling} {f.name}(
                    {", ".join(a.original_declaration for a in f.real_arguments)}) {{
            {timing_code_guest("before_marshal", str(f.name), f.generate_timing_code)}

            const int ava_is_in = 1, ava_is_out = 0;
            intptr_t __call_id = ava_get_call_id(&__ava_endpoint);

            #ifdef AVA_BENCHMARKING_MIGRATE
            if (__ava_endpoint.migration_call_id >= 0 && __call_id ==
            __ava_endpoint.migration_call_id) {{
                printf("start live migration at call_id %d\\n", __call_id);
                __ava_endpoint.migration_call_id = -2;
                start_live_migration(__chan);
            }}
            #endif

            {alloc_list.alloc}

            {"".join(compute_argument_value(a) for a in f.implicit_arguments)}

            {compute_total_size(f.arguments, lambda a: a.input)}
            struct {f.call_spelling}* __cmd = (struct {f.call_spelling}*)command_channel_new_command(
                __chan, sizeof(struct {f.call_spelling}), __total_buffer_size);
            __cmd->base.api_id = {f.api.number_spelling};
            __cmd->base.command_id = {f.call_id_spelling};
            __cmd->base.thread_id = shadow_thread_id(nw_shadow_thread_pool);
            __cmd->base.original_thread_id = __cmd->base.thread_id;

            __cmd->__call_id = __call_id;

            {nl.join(a.declaration + ";" for a in f.logue_declarations)}
            {{
                {"".join(attach_for_argument(a, "__cmd") for a in f.implicit_arguments)}
                {lines(f.prologue)}
                {"".join(attach_for_argument(a, "__cmd") for a in f.real_arguments)}
            }}

            struct {f.call_record_spelling}* __call_record =
                (struct {f.call_record_spelling}*)calloc(1, sizeof(struct {f.call_record_spelling}));
            {pack_struct("__call_record", f.arguments + f.logue_declarations, "->")}
            __call_record->__call_complete = 0;
            __call_record->__handler_deallocate = {is_async};
            ava_add_call(&__ava_endpoint, __call_id, __call_record);

            {timing_code_guest("before_send_command", str(f.name), f.generate_timing_code)}

            {send_code}

            {alloc_list.dealloc}

            {return_code}
        }}
        """


def unsupported_function_implementation(f: Function) -> str:
    """
    Generate a stub function which simply fails with an "Unsupported" message.
    :param f: The unsupported function.
    :return: A C function definition.
    """
    with location(f"at {term.yellow(str(f.name))}", f.location):
        return f"""
        EXPORTED {(f.api.export_qualifier + " ") if f.api.export_qualifier else ""}{f.return_value.type.spelling} {f.name}(
                    {", ".join(a.declaration for a in f.real_arguments)}) {{
            abort_with_reason("Unsupported API function: {f.name}");
        }}
        """


def function_wrapper(f: Function) -> str:
    """
    Generate a wrapper function for f which takes the arguments of the function, executes the "logues", and
    calls the function.
    :param f: A function.
    :return: A C static function definition.
    """
    with location(f"at {term.yellow(str(f.name))}", f.location):
        if f.return_value.type.is_void:
            declare_ret = ""
            capture_ret = ""
            return_statement = "return;"
        else:
            declare_ret = f"{f.return_value.type.nonconst.attach_to(f.return_value.name)};"
            capture_ret = f"{f.return_value.name} = "
            return_statement = f"return {f.return_value.name};"
        if f.disable_native:
            # This works for both normal functions and callbacks because the
            # difference between the two is in the call, which is not emitted in
            # this case anyway.
            capture_ret = ""
            call_code = ""
            callback_unpack = ""
        elif not f.callback_decl:
            # Normal call
            call_code = f"""({f.return_value.type.nonconst.spelling})({f.name}({", ".join(a.name for a in f.real_arguments)}))"""
            callback_unpack = ""
        else:
            # Indirect call (callback)
            try:
                (userdata_arg,) = [a for a in f.arguments if a.userdata]
            except ValueError:
                generate_requires(
                    False, "ava_callback_decl function must have exactly one argument annotated with " "ava_userdata."
                )
            call_code = f"""__target_function({", ".join(a.name for a in f.real_arguments)})"""
            callback_unpack = f"""
                {f.type.attach_to("__target_function")};
                __target_function = {f.type.cast_type(f"((struct ava_callback_user_data*){userdata_arg.name})->function_pointer")};
                {userdata_arg.name} = ((struct ava_callback_user_data*){userdata_arg.name})->userdata;
            """

        return f"""
        static {f.return_value.type.spelling} __wrapper_{f.name}({", ".join(a.declaration for a in f.arguments)}) {{
            {callback_unpack}\
            {lines(a.declaration + ";" for a in f.logue_declarations)}\
            {lines(f.prologue)}\
            {{
            {declare_ret}
            {capture_ret}{call_code};
            {lines(f.epilogue)}

            /* Report resources */
            {lines(report_alloc_resources(arg) for arg in f.arguments)}
            {report_alloc_resources(f.return_value)}
            {report_consume_resources(f)}

            {return_statement}
            }}
        }}
        """.strip()


def call_function_wrapper(f: Function) -> ExprOrStr:
    """
    Call a functions wrapper with arguments that are already in scope.
    :param f: The function to call.
    :return: A C statement to perform the call and capture the return value.
    """
    if f.return_value.type.is_void:
        capture_ret = ""
    else:
        capture_ret = f"{f.return_value.type.nonconst.attach_to(f.return_value.name)}; {f.return_value.name} = ({f.return_value.type.nonconst.spelling})"
    return f"""
        {capture_ret}__wrapper_{f.name}({", ".join(f"({a.type.spelling}){a.name}" for a in f.arguments)});
    """.strip()
