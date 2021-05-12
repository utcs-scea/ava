"""
Glossary:

The //transfer buffer// is the buffer that contains the data transported from the guest.
This may be either a pointer into shared memory or a local buffer filled from a socket.
In some cases, the transfer buffer is also used as the local buffer.

The //local buffer// is the callee local buffer containing potentially translated values taken from the transfer buffer.
The local buffer is the same as the transfer buffer if translation is not needed and other requirements are met.
"""

from typing import Optional, Union, Any, Iterable, Callable

from nightwatch import location, term
from nightwatch.c_dsl import ExprOrStr, Expr
from nightwatch.generator.c.util import compute_buffer_size, for_all_elements
from nightwatch.generator.common import comment_block, nl
from nightwatch.model import Type, Argument, ConditionalType, lines

# FIXME: Add support for NW_ZEROCOPY_BUFFER

DECLARE_BUFFER_SIZE_EXPR = Expr("volatile size_t __buffer_size = 0;")


def get_transfer_buffer_expr(value: str, type: Type, *, not_null=False) -> ExprOrStr:
    """
    Generate an expression that gets the transfer buffer of `value` (which has `type`).
    :param value: The value whose transfer buffer to get.
    :param type: The type of value.
    :param not_null: If True, then this function does not generate NULL checks.
    :return: A C expression.
    """
    return (Expr(value).not_equals("NULL") | not_null).if_then_else_expression(
        Expr(type.transfer)
        .equals("NW_BUFFER")
        .if_then_else_expression(
            f"({type.spelling})command_channel_get_buffer(__chan, __cmd, {value})",
            Expr(type.transfer)
            .equals("NW_ZEROCOPY_BUFFER")
            .if_then_else_expression(                
                # TODO(yuhc): Add back zero copy region supprot after this feature is refactored.
                # f"({type.spelling})ava_zcopy_region_decode_position_independent(__ava_endpoint.zcopy_region, {value})",
                f"({type.spelling})command_channel_get_buffer(__chan, __cmd, {value})",
                f"({type.spelling}){value}",
            ),
        ),
        f"({type.spelling}){value}",
    )


def get_buffer_expr(value: str, type: Type, *, not_null=False, size_out=None) -> Expr:
    """
    Generate an expression that finds the local buffer to use for `value`.
    :param value: The value.
    :param type: The type of value.
    :param not_null: If True, then this function does not generate NULL checks.
    :param size_out: If non-None, fill the existing variable with the buffer size in BYTES.
    :return: A C expression.
    """
    size_out = ("&" + size_out) if size_out else "NULL"
    return (type.lifetime.not_equals("AVA_CALL") & (Expr(value).not_equals("NULL") | not_null)).if_then_else_expression(
        f"""
        ({type.spelling})ava_shadow_buffer_get_buffer(&__ava_endpoint, __chan, __cmd, {value},
                {type.lifetime}, {type.lifetime_coupled},
                {size_out}, {type.buffer_allocator}, {type.buffer_deallocator})
        """,
        (
            type.lifetime.equals("AVA_CALL") & type.transfer.one_of({"NW_BUFFER", "NW_ZEROCOPY_BUFFER"})
        ).if_then_else_expression(
            get_transfer_buffer_expr(value, type, not_null=not_null), f"({type.spelling}){value}"
        ),
    )


def get_buffer(
    target: str,
    value: str,
    type: Type,
    *,
    original_type: Optional[Type],
    precomputed_size=None,
    not_null=False,
    declare_buffer_size=True,
) -> Expr:
    """
    Generate code that sets target to the local buffer to use for value.
    After this code executes `__buffer_size` is equal to the buffer size in elements.
    :param target: The variable in which to store the buffer pointer.
    :param value: The original value to translate.
    :param type: The type of value.
    :param original_type: The original type if type is immediately inside a ConditionalType.
    :param precomputed_size: The expression for a precomputed size of value in *elements*.
    :param not_null: If True, then this function does not generate NULL checks.
    :return: A series of C statements.
    """
    size_expr = compute_buffer_size(type, original_type) if precomputed_size is None else precomputed_size
    declarations = DECLARE_BUFFER_SIZE_EXPR if declare_buffer_size else Expr("")
    return declarations.then(
        (type.lifetime.not_equals("AVA_CALL") & (Expr(value).not_equals("NULL") | not_null)).if_then_else(
            # FIXME: The initial assert is probably incorrect. Should it be checking the ADDRESSES? Should it be there at all? Commented for now: AVA_DEBUG_ASSERT({target} != {value}); (was at top of code)
            # FIXME: Should we check the buffer size? Can it be computed here? AVA_DEBUG_ASSERT(__buffer_size == {size_expr}); (was at bottom of code)
            f"""
            /* Size is in bytes until the division below. */
            {target} = {get_buffer_expr(value, type, not_null=not_null, size_out="__buffer_size")};
            AVA_DEBUG_ASSERT(__buffer_size % sizeof({type.pointee.spelling}) == 0);
            __buffer_size /= sizeof({type.pointee.spelling});
            /* Size is now in elements. */
        """.lstrip(),
            f"""
            __buffer_size = {size_expr};
            {target} = {get_buffer_expr(value, type, not_null=not_null, size_out="__buffer_size")};
        """.strip(),
        )
    )


def size_to_bytes(size: ExprOrStr, type: Type) -> str:
    return f"{size} * sizeof({type.pointee.spelling})"


def attach_buffer(
    target: str,
    value: str,
    data: str,
    type: Type,
    copy: Union[Expr, Any],
    *,
    cmd,
    original_type: Optional[Type],
    expect_reply: bool,
    precomputed_size=None,
) -> Expr:
    """
    Generate code to attach a buffer to a command.

    :param target: The value to set to the buffer offset after attaching.
    :param value: The value to attach.
    :param type: The type of value.
    :param copy: An expression which is true if if this value should be copied.
    :param cmd: The command to attach to.
    :param original_type: The original type if type is immediately inside a ConditionalType.
    :param precomputed_size: The expression for a precomputed size of value in *elements*.
    :return: A series of C statements.
    """
    cmd = f"(struct command_base*){cmd}"
    size_expr = size_to_bytes(
        compute_buffer_size(type, original_type) if precomputed_size is None else precomputed_size, type
    )

    def simple_attach(func):
        return (
            lambda: f"""{target} = ({type.nonconst.spelling}){func}(__chan, {cmd}, {data}, {size_expr});
                        """
        )

    def zerocopy_attach(func="ava_zcopy_region_encode_position_independent"):
        return (
            lambda: f"""{target} = ({type.nonconst.spelling}){func}(__ava_endpoint.zcopy_region, {data});
                        """
        )

    def shadow_attach(func):
        return (
            lambda: f"""{target} = ({type.nonconst.spelling}){func}(&__ava_endpoint,
                        __chan, {cmd}, {value}, {data}, {size_expr},
                        {type.lifetime}, {type.buffer_allocator}, {type.buffer_deallocator},
                        alloca(sizeof(struct ava_buffer_header_t)));"""
        )

    return type.transfer.equals("NW_ZEROCOPY_BUFFER").if_then_else(
        # TODO(yuhc): Add back zero copy region supprot after this feature is refactored.
        # zerocopy_attach(),
        simple_attach("command_channel_attach_buffer"),
        type.lifetime.equals("AVA_CALL").if_then_else(
            Expr(copy).if_then_else(
                simple_attach("command_channel_attach_buffer"),
                f"{target} = HAS_OUT_BUFFER_SENTINEL;" if expect_reply else f"{target} = NULL; /* No output */\n",
            ),
            Expr(copy).if_then_else(
                shadow_attach("ava_shadow_buffer_attach_buffer"),
                shadow_attach("ava_shadow_buffer_attach_buffer_without_data"),
            ),
        ),
    )


def compute_total_size(args: Iterable[Argument], copy_pred: Callable[[Argument], Expr]) -> str:
    """
    Sum the sizes of all the buffers created by all arguments.
    :param args: All the arguments to sum.
    :param copy_pred: A function which returns true if the data associated with this argument will be copied.
    :return: A series of C statements.
    """
    size = "__total_buffer_size"

    def compute_size(values, type: Type, depth, argument: Argument, original_type=None, **other):
        if isinstance(type, ConditionalType):
            return Expr(type.predicate).if_then_else(
                compute_size(values, type.then_type, depth, argument, original_type=type.original_type, **other),
                compute_size(values, type.else_type, depth, argument, original_type=type.original_type, **other),
            )

        (value,) = values
        pred = Expr(type.transfer).equals("NW_BUFFER") & Expr(value).not_equals("NULL") & (Expr(type.buffer) > 0)

        def add_buffer_size():
            size_expr = size_to_bytes(compute_buffer_size(type, original_type), type)
            return Expr(copy_pred(argument)).if_then_else(
                type.lifetime.equals("AVA_CALL").if_then_else(
                    f"{size} += command_channel_buffer_size(__chan, {size_expr});\n",
                    f"{size} += ava_shadow_buffer_size(&__ava_endpoint, __chan, {size_expr});\n",
                ),
                type.lifetime.equals("AVA_CALL").if_then_else(
                    "", f"{size} += ava_shadow_buffer_size_without_data(&__ava_endpoint, __chan, {size_expr});\n"
                ),
            )

        def simple_buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            return pred.if_then_else(add_buffer_size)

        def buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            loop = for_all_elements(values, type, depth=depth, argument=argument, original_type=original_type, **other)
            outer_buffer = str(add_buffer_size())
            return pred.if_then_else(loop + outer_buffer)

        if type.fields:
            return for_all_elements(values, type, depth=depth, argument=argument, original_type=original_type, **other)
        return type.is_simple_buffer(allow_handle=True).if_then_else(
            simple_buffer_case, lambda: Expr(type.transfer).equals("NW_BUFFER").if_then_else(buffer_case)
        )

    size_code = lines(
        comment_block(
            f"Size: {a}",
            compute_size(
                (a.name,),
                a.type,
                depth=0,
                name=a.name,
                kernel=compute_size,
                only_complex_buffers=False,
                argument=a,
                self_index=0,
            ),
        )
        for a in args
    )
    return f"size_t {size} = 0;{nl}{{ {size_code} }}"


def deallocate_managed_for_argument(arg: Argument, src: str):
    def convert_result_value(values, type: Type, depth, original_type=None, **other):
        if isinstance(type, ConditionalType):
            return Expr(type.predicate).if_then_else(
                convert_result_value(values, type.then_type, depth, original_type=type.original_type, **other),
                convert_result_value(values, type.else_type, depth, original_type=type.original_type, **other),
            )

        (local_value,) = values
        buffer_pred = Expr(type.transfer).equals("NW_BUFFER") & f"{local_value} != NULL"
        dealloc_shadows = Expr(type.deallocates).if_then_else(
            f"ava_shadow_buffer_free_coupled(&__ava_endpoint, (void *){local_value});"
        )

        def simple_buffer_case():
            return ""

        def buffer_case():
            if not hasattr(type, "pointee"):
                return """abort_with_reason("Reached code to handle buffer in non-pointer type.");"""
            return buffer_pred.if_then_else(
                for_all_elements(values, type, depth=depth, original_type=original_type, **other)
            )

        def default_case():
            dealloc_code = Expr(type.deallocates).if_then_else(
                Expr(type.transfer)
                .equals("NW_HANDLE")
                .if_then_else(
                    f"""
                    ava_coupled_free(&__ava_endpoint, {local_value});
                    """.strip()
                )
            )
            return dealloc_code

        if type.fields:
            return for_all_elements(values, type, depth=depth, original_type=original_type, **other)
        return (
            type.is_simple_buffer(allow_handle=False)
            .if_then_else(
                simple_buffer_case,
                Expr(type.transfer)
                .equals("NW_BUFFER")
                .if_then_else(
                    buffer_case, (Expr(type.transfer).one_of({"NW_OPAQUE", "NW_HANDLE"})).if_then_else(default_case)
                ),
            )
            .then(dealloc_shadows)
            .scope()
        )

    with location(f"at {term.yellow(str(arg.name))}", arg.location):
        conv = convert_result_value(
            (f"""{src + "->" if src else ""}{arg.name}""",),
            arg.type,
            depth=0,
            name=arg.name,
            kernel=convert_result_value,
            self_index=0,
        )
        return comment_block(f"Dealloc: {arg}", conv)


def allocate_tmp_buffer(tmp_name: str, size_name: str, type: Type, *, alloc_list, original_type=None) -> str:
    return f"""
        const size_t {size_name} = {compute_buffer_size(type, original_type)};
        {type.nonconst.attach_to(tmp_name)};
        {tmp_name} = ({type.nonconst.spelling})calloc(1, {size_to_bytes(size_name, type)});
        {alloc_list.insert(tmp_name, "free")}
        """.strip()
