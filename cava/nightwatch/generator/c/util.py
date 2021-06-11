from typing import Optional, List
from functools import reduce
from toposort import toposort_flatten, CircularDependencyError

from nightwatch import location, term
from nightwatch.c_dsl import Expr
from nightwatch.extension import extension
from nightwatch.generator import generate_expects
from nightwatch.model import Type, Function, StaticArray, lines


@extension(Type)
class _TypeExtensions:
    # def deep_sizeof(self, inner=False):
    #     self_size = f"sizeof({self.spelling})"
    #     if self.buffer:
    #         return f"""{self_size + " + " if inner else ""} ({self.buffer}) * ({self.pointee.deep_sizeof(True)})"""
    #     else:
    #         return self_size

    def is_blob(self, allow_handle=False):
        """
        Check if self is a blob type.
        A blob is a type that can be treated as a sequence of bytes without any semantics.
        This implies that no elements within the blob need to be translated in any way
        during command processing.
        :param allow_handle: If True, allow handles inside the blob.
        :return: True, iff self is a blob.
        """
        individual_field_preds = [t.is_blob(allow_handle) for t in self.fields.values()]
        fields_pred = reduce(lambda a, b: Expr(a) & b, individual_field_preds, True)
        allowed = {"NW_OPAQUE", "NW_CALLBACK", "NW_CALLBACK", "NW_CALLBACK_REGISTRATION"} | (
            {"NW_HANDLE"} if allow_handle else set()
        )
        transfer_pred = Expr(self.transfer).one_of(allowed)
        pointee_pred = True
        if isinstance(self, StaticArray):
            pointee_pred = self.pointee.is_blob(allow_handle) & Expr(self.buffer_allocator == "malloc")
        pred = transfer_pred & fields_pred & pointee_pred
        if "ava_index" in str(pred):  # TODO: Remove this hack by tracking the usage of variables in expressions.
            return Expr(False)
        return pred

    def is_simple_buffer(self, allow_handle=False):
        if hasattr(self, "pointee") and self.pointee and not isinstance(self, StaticArray):
            return self.pointee.is_blob(allow_handle) & self.transfer.one_of({"NW_BUFFER", "NW_ZEROCOPY_BUFFER"})
        return Expr(False)


_letters = "abcdefghikmnoprstuwxyz"


def _remove_provided_arguments(type_):
    d = type_.__dict__.copy()
    d.pop("spelling", None)
    return d


def _char_type_like(type_):
    return Type("char", **_remove_provided_arguments(type_))


def compute_buffer_size(type_: Type, original_type: Optional[Type] = None):
    size_expr = Expr(type_.buffer)
    if original_type and original_type.pointee.spelling != type_.pointee.spelling:
        pointee_size = f"sizeof({type_.pointee.spelling})"
        # for void* buffer, assume that each element is 1 byte
        if type_.pointee.is_void:
            pointee_size = "1"
        size_adjustment = f" * sizeof({original_type.pointee.spelling}) / {pointee_size}"
    else:
        size_adjustment = ""
    return Expr(f"(size_t){size_expr.group()}{size_adjustment}").group()


def for_all_elements(
    values: tuple,
    # pylint: disable=unused-argument
    cast_type: Type,
    type_: Type,
    *,
    depth: int,
    kernel,
    name: str,
    self_index: int,
    precomputed_size=None,
    original_type=None,
    **extra,
):
    """
    kernel(values, cast_type, type, **other)
    """
    size = f"__{name}_size_{depth}"
    index = f"__{name}_index_{depth}"

    with location(f"in type {term.yellow(type_.spelling)}"):
        if hasattr(type_, "pointee") and type_.pointee:
            loop = ""
            size_expr = Expr(precomputed_size or compute_buffer_size(type_, original_type))
            eval_size = f"const size_t {size} = {size_expr};"
            inner_values = tuple(f"__{name}_{_letters[i]}_{depth}" for i in range(len(values)))
            type_pointee = _char_type_like(type_.pointee) if type_.pointee.is_void else type_.pointee
            nested = kernel(
                tuple("*" + v for v in inner_values),
                type_pointee.nonconst,
                type_pointee,
                depth=depth + 1,
                name=name,
                kernel=kernel,
                self_index=self_index,
                **extra,
            )
            if nested:
                set_inner_values = lines(
                    f"""
                     {type_pointee.nonconst.attach_to(iv, additional_inner_type_elements="*")};
                     {iv} = {type_pointee.nonconst.cast_type(type_.ascribe_type(v), "*")} + {index};
                     """
                    for v, iv in zip(values, inner_values)
                )
                if size_expr.is_constant(1):
                    loop = f"""
                        const size_t {index} = 0;
                        const size_t ava_index = 0;
                        {set_inner_values}
                        {nested}
                    """
                else:
                    loop = f"""
                        for(size_t {index} = 0; {index} < {size}; {index}++) {{
                            const size_t ava_index = {index};
                            {set_inner_values}
                            {nested}
                        }}
                    """.strip()

            if nested:
                return eval_size + loop
            return ""

        if type_.fields:
            prefix = f"""
                {type_.nonconst.attach_to("ava_self", additional_inner_type_elements="*")};
                ava_self = {type_.nonconst.cast_type(type_.ascribe_type(f"&{values[self_index]}", "*"), "*")};
            """
            field_infos = []
            for field_name, field in type_.fields.items():
                inner_values = tuple(f"__{name}_{_letters[i]}_{depth}_{field_name}" for i in range(len(values)))
                nested = kernel(
                    tuple("*" + v for v in inner_values),
                    field.nonconst,
                    field,
                    depth=depth + 1,
                    name=name,
                    kernel=kernel,
                    self_index=self_index,
                    **extra,
                )
                inner_code = ""
                if str(nested).strip():
                    set_inner_values = lines(
                        f"""
                        {field.nonconst.attach_to(iv, additional_inner_type_elements="*")};
                        {iv} = {field.nonconst.cast_type(field.ascribe_type(f"&({v}).{field_name}", "*"), "*")};
                        """.strip()
                        for v, iv in zip(values, inner_values)
                    )
                    inner_code = set_inner_values + str(nested)
                field_infos.append((field_name, nested, inner_code))
            code = "\n".join(inner_code for _, _, inner_code in _sort_fields(field_infos))
            if code.strip():
                return "{" + prefix + code + "}"
            return ""

        raise ValueError("Type must be a buffer of some kind.")


def _sort_fields(field_infos: List[tuple]):
    dag = {}
    fields = {name: (name, k, c) for name, k, c in field_infos}
    # TODO: Sort based on simple string containment.
    for name, kernel, _ in field_infos:
        dag[name] = set(n for n in fields.keys() if n in str(kernel))
    # Compute an order which honors the deps
    try:
        return [fields[n] for n in toposort_flatten(dag, sort=True)]
    except CircularDependencyError:
        generate_expects(
            False, "Struct fields may be processed out of order, due to hacks in the current implementation."
        )
        return field_infos


class AllocList:
    def __init__(self, f: Function):
        self.name = f"__ava_alloc_list_{f.name}"
        # The estimate is currently zero since this totally avoids allocating the ptr array in cases where it isn't
        # needed.
        self.reserve = 0

    @property
    def alloc(self):
        return f"""
            GPtrArray *{self.name} = g_ptr_array_new_full({self.reserve}, (GDestroyNotify)ava_buffer_with_deallocator_free);
        """.strip()

    def insert(self, ptr, deallocator):
        return f"""
            g_ptr_array_add({self.name}, ava_buffer_with_deallocator_new({deallocator}, {ptr}));
        """.strip()

    @property
    def dealloc(self):
        return f"""
            g_ptr_array_unref({self.name}); /* Deallocate all memory in the alloc list */
        """.strip()
