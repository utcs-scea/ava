from typing import Optional

from nightwatch import location, term
from nightwatch.c_dsl import Expr, ExprOrStr
from nightwatch.generator.c.buffer_handling import get_transfer_buffer_expr, get_buffer, attach_buffer, \
    compute_total_size, deallocate_managed_for_argument, size_to_bytes, allocate_tmp_buffer
from nightwatch.generator.c.instrumentation import timing_code_worker
from nightwatch.generator.c.stubs import call_function_wrapper
from nightwatch.generator.c.util import AllocList, compute_buffer_size, for_all_elements
from nightwatch.generator.common import comment_block, lines
from nightwatch.model import Argument, Type, ConditionalType, Function

log_call_declaration = "ssize_t __call_log_offset = -1;"
log_ret_declaration = "ssize_t __ret_log_offset = -1;"


def convert_input_for_argument(arg: Argument, src):
    """
    Generate code to extract the value for arg from the call structure in src.
    The value of arg is left in a variable named arg.name. The value is fully
    converted to local values. This code used in the command receiver to
    implement a CALL command.
    :param arg: The argument to extract.
    :param src: The CALL command structure.
    :return: A series of C statements to perform the extraction.
    """
    alloc_list = AllocList(arg.function)

    def convert_input_value(values, type: Type, depth, original_type=None, **other):
        local_value, param_value = values
        preassignment = f"{local_value} = {get_transfer_buffer_expr(param_value, type)};"

        if isinstance(type, ConditionalType):
            return Expr(preassignment).then(Expr(type.predicate).if_then_else(
                convert_input_value(values, type.then_type, depth, original_type=type.original_type, **other),
                convert_input_value(values, type.else_type, depth, original_type=type.original_type, **other)))

        if type.is_void:
            return """abort_with_reason("Reached code to handle void value.");"""

        def maybe_alloc_local_temporary_buffer():
            # TODO: Deduplicate with allocate_tmp_buffer
            allocator = type.buffer_allocator
            deallocator = type.buffer_deallocator
            return Expr(param_value).not_equals("NULL").if_then_else(f"""{{
            const size_t __size = {compute_buffer_size(type, original_type)};                                   
            {local_value} = ({type.nonconst.spelling}){allocator}({size_to_bytes("__size", type)});    
            {alloc_list.insert(local_value, deallocator)}
            }}""")

        src_name = f"__src_{arg.name}_{depth}"

        def get_buffer_code():
            return f"""
                {type.nonconst.attach_to(src_name)};
                {src_name} = {local_value};
                {get_buffer(local_value, param_value, type, original_type=original_type, not_null=True)}
                {(type.lifetime.equals("AVA_CALL") & (~type.is_simple_buffer() | type.buffer_allocator.not_equals("malloc"))).if_then_else(
                        maybe_alloc_local_temporary_buffer)}
                """

        def simple_buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            copy_code = (Expr(arg.input) & Expr(local_value).not_equals(src_name)).if_then_else(
                f"""memcpy({local_value}, {src_name}, {size_to_bytes("__buffer_size", type)});""")
            return ((type.lifetime.not_equals("AVA_CALL") | arg.input) & Expr(param_value).not_equals("NULL")).if_then_else(
                f"""
                    {get_buffer_code()}
                    {copy_code}
                """.strip(),
                (Expr(arg.input) | type.transfer.equals("NW_ZEROCOPY_BUFFER")).if_then_else(
                    preassignment,
                    maybe_alloc_local_temporary_buffer
                )
            )

        def buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            if not arg.input:
                return simple_buffer_case()

            inner_values = (local_value, src_name)
            core = for_all_elements(inner_values, type, depth=depth,
                                    precomputed_size="__buffer_size", original_type=original_type, **other)
            return ((type.lifetime.not_equals("AVA_CALL") | arg.input) & Expr(param_value).not_equals("NULL")).if_then_else(
                        f"""
                        {get_buffer_code()}
                        {core}
                        """.strip(),
                        maybe_alloc_local_temporary_buffer
                    )

        def default_case():
            def deref_code(handlepool_function: str) -> callable:
                return lambda: (Expr(type.transfer).one_of({"NW_CALLBACK", "NW_CALLBACK_REGISTRATION"})).if_then_else(
                    f"{local_value} =  ({param_value} == NULL) ? NULL : {type.callback_stub_function};",
                    (Expr(type.transfer).equals("NW_HANDLE")).if_then_else(
                        f"{local_value} = ({type.nonconst.spelling}){handlepool_function}(handle_pool, (void*){param_value});",
                    Expr(not type.is_void).if_then_else(
                        f"{local_value} = {param_value};",
                        """abort_with_reason("Reached code to handle void value.");"""
                    )))

            return Expr(type.deallocates).if_then_else(
                    deref_code("nw_handle_pool_deref_and_remove"),
                    deref_code("nw_handle_pool_deref"))

        if type.fields:
            return for_all_elements(values, type, depth=depth, **other)
        rest = type.is_simple_buffer().if_then_else(
            simple_buffer_case,
            Expr(type.transfer).equals("NW_BUFFER").if_then_else(
                buffer_case,
                default_case
            )
        )
        if rest:
            return Expr(preassignment).then(rest).scope()
        else:
            return ""

    with location(f"at {term.yellow(str(arg.name))}", arg.location):
        conv = convert_input_value((arg.name, f"{src}->{arg.param_spelling}"), arg.type,
                                   depth=0, name=arg.name,
                                   kernel=convert_input_value,
                                   original_type=arg.type, self_index=0)
        return comment_block(f"Input: {arg}", f"""\
        {arg.type.nonconst.attach_to(arg.name)}; \
        {conv}
        """)


def convert_result_for_argument(arg: Argument, dest) -> ExprOrStr:
    """
    Take the value of arg in the local scope and write it into dest.
    :param arg: The argument to place in the output.
    :param dest: A RET command struct pointer.
    :return: A series of C statements.
    """
    alloc_list = AllocList(arg.function)

    def convert_result_value(values, type: Type, depth, original_type=None, **other) -> str:
        if isinstance(type, ConditionalType):
            return Expr(type.predicate).if_then_else(
                convert_result_value(values, type.then_type, depth, original_type=type.original_type, **other),
                convert_result_value(values, type.else_type, depth, original_type=type.original_type, **other))

        if type.is_void:
            return """abort_with_reason("Reached code to handle void value.");"""

        param_value, local_value = values

        def attach_data(data):
            return attach_buffer(param_value, local_value, data, type, arg.output, cmd=dest, original_type=original_type, expect_reply=False)

        def simple_buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            return (Expr(local_value).not_equals("NULL") & (Expr(type.buffer) > 0)).if_then_else(
                attach_data(local_value),
                f"{param_value} = NULL;")

        def buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            if not arg.output:
                return simple_buffer_case()

            tmp_name = f"__tmp_{arg.name}_{depth}"
            size_name = f"__size_{arg.name}_{depth}"
            inner_values = (tmp_name, local_value)
            return Expr(local_value).not_equals("NULL").if_then_else(
                f"""{{
                {allocate_tmp_buffer(tmp_name, size_name, type, alloc_list=alloc_list, original_type=original_type)}
                {for_all_elements(inner_values, type, precomputed_size=size_name, depth=depth, original_type=original_type, **other)}
                {attach_data(tmp_name)}
                }}""",
                f"{param_value} = NULL;")

        def default_case():
            handlepool_function = "nw_handle_pool_lookup_or_insert"
            return Expr(type.transfer).equals("NW_HANDLE").if_then_else(
                Expr(type.deallocates).if_then_else(
                    f"{param_value} = NULL;",
                    f"{param_value} = ({type.nonconst.spelling}){handlepool_function}(handle_pool, (void*){local_value});"),
                Expr(not type.is_void).if_then_else(
                    f"{param_value} = {local_value};")
            )

        if type.fields:
            return for_all_elements(values, type, depth=depth, original_type=original_type, **other)
        return type.is_simple_buffer().if_then_else(
            simple_buffer_case,
            Expr(type.transfer).equals("NW_BUFFER").if_then_else(
                buffer_case,
                default_case
            )
        ).scope()

    with location(f"at {term.yellow(str(arg.name))}", arg.location):
        conv = convert_result_value((f"{dest}->{arg.param_spelling}", f"{arg.name}"), arg.type,
                                    depth=0, name=arg.name,
                                    kernel=convert_result_value, self_index=1)
        return (Expr(arg.output) | arg.ret).if_then_else(
            comment_block(f"Output: {arg}", conv))


def call_command_implementation(f: Function):
    with location(f"at {term.yellow(str(f.name))}", f.location):
        alloc_list = AllocList(f)

        is_async = ~Expr(f.synchrony).equals("NW_SYNC");
        reply_code = f"""
            command_channel_send_command(__chan, (struct command_base*)__ret);
        """.strip()

        if (f.api.reply_code):
            import_code = f.api.reply_code.encode('ascii', 'ignore').decode('unicode_escape')[1:-1]
            ldict = locals()
            exec(import_code, globals(), ldict)
            reply_code = ldict['reply_code']

        worker_argument_process_code = ""
        if (f.api.worker_argument_process_code):
            import_code = f.api.worker_argument_process_code.encode('ascii', 'ignore').decode('unicode_escape')[1:-1]
            ldict = locals()
            exec(import_code, globals(), ldict)
            worker_argument_process_code = ldict['worker_argument_process_code']

        return f"""
        case {f.call_id_spelling}: {{\
            {timing_code_worker("before_unmarshal", str(f.name), f.generate_timing_code)}
            ava_is_in = 1; ava_is_out = 0;
            {alloc_list.alloc}
            struct {f.call_spelling}* __call = (struct {f.call_spelling}*)__cmd;
            assert(__call->base.api_id == {f.api.number_spelling});
            assert(__call->base.command_size == sizeof(struct {f.call_spelling}) && "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */
            {lines(convert_input_for_argument(a, "__call") for a in f.arguments)}

            {timing_code_worker("after_unmarshal", str(f.name), f.generate_timing_code)}
            /* Perform Call */
            {worker_argument_process_code}
            {call_function_wrapper(f)}
            {timing_code_worker("after_execution", str(f.name), f.generate_timing_code)}

            ava_is_in = 0; ava_is_out = 1;
            {compute_total_size(f.arguments + [f.return_value], lambda a: a.output)}
            struct {f.ret_spelling}* __ret = (struct {f.ret_spelling}*)command_channel_new_command(
                __chan, sizeof(struct {f.ret_spelling}), __total_buffer_size);
            __ret->base.api_id = {f.api.number_spelling};
            __ret->base.command_id = {f.ret_id_spelling};
            __ret->base.thread_id = __call->base.original_thread_id;
            __ret->__call_id = __call->__call_id;

            {convert_result_for_argument(f.return_value, "__ret") if not f.return_value.type.is_void else ""}
            {lines(convert_result_for_argument(a, "__ret") for a in f.arguments if a.type.contains_buffer)}

            #ifdef AVA_RECORD_REPLAY
            {log_call_declaration}
            {log_ret_declaration}
            {lines(
            record_argument_metadata(a, src="__ret" if a.type.contains_buffer else "__call") for a in f.arguments)}
            {record_argument_metadata(f.return_value, "__ret") if not f.return_value.type.is_void else ""}
            {record_call_metadata("NULL", None) if f.object_record else ""}
            #endif

            {timing_code_worker("after_marshal", str(f.name), f.generate_timing_code)}
            /* Send reply message */
            {reply_code}
            {alloc_list.dealloc}
            {lines(deallocate_managed_for_argument(a, "") for a in f.arguments)}
            break;
        }}
        """.strip()


def record_call_metadata(handle, type: Optional[Type]):
    log_call_command = f"""if(__call_log_offset == -1) {{
        __call_log_offset = 
            command_channel_log_transfer_command(__log, __chan, (const struct command_base *)__cmd);
    }}
    assert(__call_log_offset != -1);"""
    log_ret_command = f"""if(__ret_log_offset == -1) {{
        __ret_log_offset = 
            command_channel_log_transfer_command(__log, __chan, (const struct command_base *)__ret);
    }}
    assert(__ret_log_offset != -1);"""
    def dep(dependent, dependency): return f"ava_add_dependency(&__ava_endpoint, {dependent}, {dependency});"
    return Expr(type is None or type.object_record).if_then_else(f"""
        {log_call_command}{log_ret_command}
        ava_add_recorded_call(&__ava_endpoint, {handle}, ava_new_offset_pair(__call_log_offset, __ret_log_offset));
        """).then(
        lines(dep(handle, h) for h in (type.object_depends_on if type else []))
    )


def record_argument_metadata(arg: Argument, src):
    def convert_result_value(values, type: Type, depth, original_type=None, **other) -> str:
        if isinstance(type, ConditionalType):
            return Expr(type.predicate).if_then_else(
                convert_result_value(values, type.then_type, depth, original_type=type.original_type, **other),
                convert_result_value(values, type.else_type, depth, original_type=type.original_type, **other))

        if type.is_void:
            return """abort_with_reason("Reached code to handle void value.");"""

        param_value, = values
        buffer_pred = (Expr(type.transfer).equals("NW_BUFFER") & Expr(param_value).not_equals("NULL"))

        def simple_buffer_case():
            return ""

        def buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            tmp_name = f"__tmp_{arg.name}_{depth}"
            inner_values = (tmp_name,)
            loop = for_all_elements(inner_values, type, depth=depth, original_type=original_type, **other)
            if loop:
                return buffer_pred.if_then_else(
                    f"""
                     {type.nonconst.attach_to(tmp_name)}; 
                     {tmp_name} = {param_value};
                     {loop}
                    """)
            else:
                return ""

        def default_case():
            return (Expr(type.transfer).equals("NW_HANDLE")).if_then_else(
                Expr(not type.deallocates).if_then_else(
                    assign_record_replay_functions(param_value, type).then(
                        record_call_metadata(param_value, type)),
                    expunge_calls(param_value, type)
                )
            )

        if type.fields:
            return for_all_elements(values, type, depth=depth, original_type=original_type, **other)
        return type.is_simple_buffer().if_then_else(
            simple_buffer_case,
            Expr(type.transfer).equals("NW_BUFFER").if_then_else(
                buffer_case,
                default_case
            )
        )

    with location(f"at {term.yellow(str(arg.name))}", arg.location):
        conv = convert_result_value((f"{arg.name}",), arg.type,
                                    depth=0, name=arg.name,
                                    kernel=convert_result_value, self_index=0)
        return conv


def assign_record_replay_functions(local_value, type: Type) -> Expr:
    extract: Expr = type.object_explicit_state_extract
    replace: Expr = type.object_explicit_state_replace
    return (extract.not_equals("NULL") | replace.not_equals("NULL")).if_then_else(
        Expr(f"ava_assign_record_replay_functions(&__ava_endpoint, (void*){local_value}, {extract}, {replace});"))


def expunge_calls(handle, type: Optional[Type]) -> str:
    return f"""
        ava_expunge_recorded_calls(&__ava_endpoint, __log, {handle});
        """
