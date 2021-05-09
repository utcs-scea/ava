import ast as pyast

# import astor

from ...model import *


class Name:
    __slots__ = ("names",)

    def __init__(self, v):
        def conv(v):
            if isinstance(v, pyast.Name):
                return [v.id]
            elif isinstance(v, pyast.Attribute):
                return conv(v.value) + [v.attr]
            else:
                assert isinstance(v, str)
                return v.split(".")

        self.names = tuple(conv(v))

    def __str__(self):
        return ".".join(self.names)

    def __repr__(self):
        return f'Name("{".".join(self.names)}")'

    def __add__(self, o):
        return str(self) + o

    def __radd__(self, o):
        return o + str(self)

    def __hash__(self):
        return hash(self.names)
