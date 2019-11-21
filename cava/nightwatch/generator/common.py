from typing import Iterable

from ..model import *
from ..extension import extension


def comment_block(comment, block):
    if hasattr(block, "strip") and block.strip() or block:
        return f"\n /* {comment} */\n{block}"
    return ""


nl = "\n"
snl = "\\n"


def lines(strs: Iterable, nl="\n"):
    return nl.join(str(s) for s in strs if s)


def unpack_struct(struct: str, fields: Iterable, access=".", convert=lambda v, t: v, nl="\n") -> str:
    """
    Generate statements to unpack the fields of struct into scope.

    fields: an iterable of values with attributes "name" and "type" (usually Argument).
    """
    # {f.name} = {f.type.nonconst.cast_type(f.type.ascribe_type(convert(struct + access + f.name, f.type)))};
    return lines(
        (f"""
         {f.type.nonconst.attach_to(f.name)}; 
         {f.name} = {convert(struct + access + f.name, f.type)};
         """
         for f in fields), nl=nl)


def unpack_struct_scope(code, *args, nl="\n", **kws):
    return f"""({{{nl}{unpack_struct(*args, **kws, nl=nl)}{nl}{code}{nl}}})"""


def pack_struct(struct, fields, access=".", convert=(lambda v, t: v), nl="\n") -> str:
    """
    Generate statements to pack the fields of struct from the current scope.

    fields: an iterable of values with attributes "name" and "type" (usually Argument).
    """
    # {struct}{access}{f.name} = {f.type.nonconst.cast_type(f.type.ascribe_type(convert(f.name, f.type)))};
    return lines((f"""
        {struct}{access}{f.name} = {convert(f.name, f.type)};
        """ for f in fields), nl)


# Extensions to model

@extension(Type)
class _TypeSpelling:
    # Information

    @property
    def contains_buffer(self):
        return hasattr(self, "buffer") and bool(self.buffer)

    # Identifiers

    @property
    def identifier_spelling(self):
        return identifier_spelling(self.spelling)


@extension(Argument)
class _ArgumentSpelling:
    # Information

    # Identifiers
    @property
    def param_spelling(self):
        return "{}".format(self.name)


@extension(Function)
class _FunctionSpelling:
    # Information

    # Identifiers

    @property
    def call_id_spelling(self):
        return "CALL_{}_{}".format(self.api.identifier.upper(), uncamel(self.name).upper())

    @property
    def ret_id_spelling(self):
        return "RET_{}_{}".format(self.api.identifier.upper(), uncamel(self.name).upper())

    @property
    def call_spelling(self):
        return "{}_{}_call".format(self.api.identifier.lower(), uncamel(self.name))

    @property
    def ret_spelling(self):
        return "{}_{}_ret".format(self.api.identifier.lower(), uncamel(self.name))

    @property
    def call_record_spelling(self):
        return "{}_{}_call_record".format(self.api.identifier.lower(), uncamel(self.name))


@extension(API)
class _APISpelling:
    # Filenames

    @property
    def source_extension(self):
        if self.cplusplus:
            return "cpp"
        else:
            return "c"

    @property
    def c_header_spelling(self):
        return "{}_nw.h".format(self.identifier.lower())

    @property
    def c_utilities_header_spelling(self):
        return "{}_nw_utilities.h".format(self.identifier.lower())

    @property
    def c_utility_types_header_spelling(self):
        return "{}_nw_utility_types.h".format(self.identifier.lower())

    @property
    def c_types_header_spelling(self):
        return "{}_nw_types.h".format(self.identifier.lower())

    @property
    def c_library_spelling(self):
        return "{}_nw_guestlib.{}".format(self.identifier.lower(), self.source_extension)

    @property
    def c_driver_spelling(self):
        return "{}_nw_guestdrv.{}".format(self.identifier.lower(), self.source_extension)

    @property
    def c_worker_spelling(self):
        return "{}_nw_worker.{}".format(self.identifier.lower(), self.source_extension)

    @property
    def py_library_spelling(self):
        # This cannot be renamed with "_nw_guestlib" because python will look up this file by name.
        return "{}.py".format(self.identifier.lower())

    # Identifiers

    @property
    def number_spelling(self):
        return "{}_API".format(self.identifier.upper())

    @property
    def functions_enum_spelling(self):
        return "{}_functions".format(self.identifier.lower())

    @property
    def metadata_struct_spelling(self):
        return "{}_metadata".format(self.identifier.lower())

    @property
    def ioctl_spelling(self):
        return "IOCTL_{}_CMD".format(self.identifier.upper())

    @property
    def worker_spelling(self):
        return "{}_worker".format(self.identifier.lower())

    @property
    def handle_call_spelling(self):
        return "{}_handle_call".format(self.identifier.lower())
