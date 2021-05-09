from ...model import *
from ..common import *
from .. import *

# The guest lib will probably need to import the real library (using
# import tricks). This is to allow data types and classes to be
# available in both the guest and host and allows pickling to work.
# However the guest version will need to have extensive changes, most
# classes will need to be replaced or modified and stubs will need to
# be injected.


def generate_pickler(api):
    handlers = f"""
        if isinstance(obj, {tpe}):
            apply the appropriate rule.
            This will
"""
    return f"""
class {api.identifier}Pickler(Pickler):
    def persistent_id(self, obj):
        {handlers}
        return None
"""


def generate_unpickler(api):
    return f"""
class {api.identifier}Unpickler(Unpickler):
    def persistent_load(self, id):
        ???
"""


def generate_stub(f):
    """The stub will need to apply special processing to input and return
    values to allow the same type to be processed differently
    depending on the function (or even value-level information like
    size).

    Maybe only based on type and value. No "contextual" information.
    But this would only be used for value transport, NOT for access
    control or rate-limiting. This would make nested object handling
    consistent with direct handling, but would also prevent special
    cases where a value should be read to the guest even if it is
    large and usually remote (like a tensor). For tensors we could
    draw the line along the device boundary based on metadata about
    the tensor or even just looking at values in it. This would
    dramatically penalize any cases were a CPU tensor is tightly
    coupled with the tensorflow API.

    """
    return f"""
{f.name} = {f}
"""


def source(api, errors):
    imports = lines(f"import {mod}" + (f" as {name}" if name else "") for mod, name in api.include.items())
    stubs = lines(generate_stub(f) for f in api.functions)
    return ("nw.py", imports + "\n" + stubs + generate_pickler(api) + generate_unpickler(api))
