from inspect import isclass
from typing import Callable


def replace(meth: Callable) -> Callable:
    meth.__replace__ = True
    return meth


def extension(cls) -> Callable:
    def decorator(val):
        if isclass(val):
            for n, v in val.__dict__.items():
                if n not in ("__dict__", "__module__", "__name__", "__weakref__", "__doc__"):
                    assert hasattr(v, "__replace__") or n not in cls.__dict__, n
                    setattr(cls, n, v)
        else:
            setattr(cls, val.__name__, val)
        return val

    return decorator
