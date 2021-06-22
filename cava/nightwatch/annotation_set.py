from collections import namedtuple

from nightwatch import c_dsl
from nightwatch.parser import parse_requires

default_annotations = dict(
    depends_on=set(),
    object_depends_on=set(),
    object_record=False,
    object_explicit_state_replace=c_dsl.Expr("NULL"),
    object_explicit_state_extract=c_dsl.Expr("NULL"),
    buffer_allocator=c_dsl.Expr("malloc"),
    buffer_deallocator=c_dsl.Expr("free"),
    input=False,
    output=False,
    allocates=False,
    deallocates=False,
    transfer=c_dsl.Expr("NW_OPAQUE"),
    buffer=0,
    type_cast=None,
    unsupported=False,
    userdata=False,
    callback_stub_function=c_dsl.Expr("NULL"),
    generate_timing_code=False,
    generate_stats_code=False,
    lifetime=c_dsl.Expr("AVA_CALL"),
    lifetime_coupled=c_dsl.Expr("NULL"),
    disable_native=False,
)

combinable_annotations = dict(
    # True means that if-then-else should be ignored for this annotation.
    depends_on=True,
    unsupported=True,
    object_depends_on=True,
    object_record_for=True,
)


class Conditional(namedtuple("Conditional", ["predicate", "then_branch", "else_branch"])):
    pass


class AnnotationSet(dict):
    def __init__(self, defaults, **kwds):
        super().__init__(**kwds)
        self.defaults = defaults

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            if self.defaults:
                return self.defaults[name[-1] if isinstance(name, tuple) else name]
            raise

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __setitem__(self, name, v):
        tail_name = name[-1] if isinstance(name, tuple) else name
        if tail_name in combinable_annotations:
            t = self[name]
            if isinstance(t, bool):
                v = t or v
            elif hasattr(t, "__or__"):
                v = t | v
            elif hasattr(t, "__add__"):
                v = t + v
            super().__setitem__(name, v)
        else:
            parse_requires(name not in self, "Metadata and annotations can only be provided once.")
            super().__setitem__(name, v)

    def if_else(self, predicate, else_branch):
        """
        Combine the metadata fields of self with else_branch using predicate.
        """
        assert self.defaults == else_branch.defaults
        ret = AnnotationSet(defaults=self.defaults)
        for name in set(self.keys()) | set(else_branch.keys()):
            tail_name = name[-1] if isinstance(name, tuple) else name
            if tail_name == "type_cast":
                ret[name] = Conditional(predicate, self[name], else_branch[name])
            elif tail_name in combinable_annotations and combinable_annotations[tail_name]:
                ret[name] = self[name]
                ret[name] = else_branch[name]
            else:
                ret[name] = c_dsl.Expr(predicate).if_then_else_expression(self[name], else_branch[name])
        return ret

    def pushdown(self, subelement):
        ret = AnnotationSet(defaults=self.defaults)
        for name, value in self.items():
            if isinstance(name, tuple):
                name = (subelement,) + name
            else:
                name = (subelement, name)
            ret[name] = value
        return ret

    def subelement(self, subelement):
        ret = AnnotationSet(defaults=self.defaults)
        for name, value in self.items():
            if isinstance(name, tuple) and len(name) > 1 and name[0] == subelement:
                subname = name[1:]
                if len(subname) == 1:
                    subname = subname[0]
                ret[subname] = value
        return ret

    def direct(self, only=None):
        if only:
            defaults = {}
            for name, value in self.defaults.items():
                if name in only:
                    defaults[name] = value
        else:
            defaults = self.defaults
        ret = AnnotationSet(defaults=defaults)
        for name, value in self.items():
            if not isinstance(name, tuple) and (not only or name in only):
                # parse_assert(not only or name in only, f"Unknown annotation {name}")
                ret[name] = value
        return ret

    def flatten(self):
        """
        Convert self to a normal dict.
        """
        ret = dict(self.defaults)
        ret.update(self)
        return ret

    def update(self, m) -> None:
        if hasattr(m, "items"):
            m = m.items()
        for k, v in m:
            self[k] = v


def annotation_set() -> AnnotationSet:
    return AnnotationSet(default_annotations)
