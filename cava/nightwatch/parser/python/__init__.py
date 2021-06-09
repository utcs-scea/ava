from .. import *
from ...model import *
from .model_ext import *

# import importlib.util
import ast as pyast
import astor

top_type = Opaque("object")
tuple_type = Opaque("tuple")
immutable_dict_type = Opaque("dict", out=None)


class Converter:
    def __init__(
        self,
    ):
        self.name = ""
        self.version = ""
        self.identifier = ""
        self.number = ""
        self.imports = []
        self.type_annotations = {}

    def convert(self, ast, filename):
        self.filename = filename
        decls = list(self.convert_decls(ast.body))
        print(self.type_annotations)
        return API(
            self.name,
            self.version,
            self.identifier,
            self.number,
            self.imports,
            functions=decls,
            dynamic_rules=self.type_annotations,
        )

    def convert_location(self, e):
        return Location(self.filename, getattr(e, "lineno"), getattr(e, "col_offset"), None)

    def convert_arguments(self, args):
        for a in args.args:
            yield Argument(Name(a.arg), self.convert_type(a.annotation))
        if args.vararg:
            yield Argument(Name(args.vararg.arg), tuple_type)
        if args.kwarg:
            yield Argument(Name(args.kwarg.arg), immutable_dict_type)

    def convert_annotation(self, e):
        if isinstance(e, pyast.Name):
            return {e.id: True}
        elif isinstance(e, pyast.Call):
            return {e.func: [self.convert_annotation(x) for x in e.args]}
        elif hasattr(e, "elts"):
            return self.convert_annotation(e.elts)
        else:
            try:
                ret = {}
                for x in e:
                    ret.update(self.convert_annotation(x))
                return ret
            except TypeError as e:
                parse_requires(False, f"Unsupported annotation: {e}")

    def convert_type(self, e):
        if isinstance(e, pyast.Name):
            return Opaque(e.id)
        elif isinstance(e, pyast.Subscript):
            parse_requires(e.value.attr == "a", "Subscripting may only be used on '.a' for applying annotations")
            annotations = self.convert_annotation(e.slice.value)
            return Opaque(Name(e.value.value), **annotations)
        elif isinstance(e, pyast.BinOp):
            annotations = self.convert_annotation(e.right)
            return Opaque(Name(e.left), **annotations)
        elif e is None:
            return top_type
        else:
            parse_requires(False, "Unsupported type")

    def convert_decls(self, exprs, prefix=None):
        is_toplevel = prefix is None
        ret = []
        for e in exprs:
            name = None
            if hasattr(e, "name"):
                name = pyast.Name(e.name, pyast.Load) if not prefix else pyast.Attribute(prefix, e.name, pyast.Load)
            with location(astor.to_source(name) if name else "", loc=self.convert_location(e)):
                if isinstance(e, pyast.FunctionDef):
                    args = list(self.convert_arguments(e.args))
                    function_annotations = self.convert_annotation(e.decorator_list)
                    f = Function(
                        Name(name),
                        Argument(Name(RET_ARGUMENT_NAME), self.convert_type(e.returns)),
                        args,
                        **function_annotations,
                        location=self.convert_location(e),
                    )
                    yield f
                elif isinstance(e, pyast.ClassDef):
                    for ann in e.decorator_list:
                        f = ann.func
                        if hasattr(f, "id") and f.id == "nw":
                            name = ann.args[0]
                    yield from self.convert_decls(e.body, name)
                elif isinstance(e, pyast.Assign):
                    parse_requires(is_toplevel, "Global configuration can only be set at the top level.")
                    t = e.targets[0]
                    parse_requires(isinstance(t, pyast.Name), "Global assignments can only create new constants.")
                    self.__dict__[t.id] = pyast.literal_eval(e.value)
                elif isinstance(e, pyast.Expr):
                    e = e.value
                    if isinstance(e, pyast.Subscript):
                        parse_requires(
                            e.value.attr == "a", "Subscripting may only be used on '.a' for applying annotations"
                        )
                        annotations = self.convert_annotation(e.slice.value)
                        self.type_annotations[Name(e.value.value)] = annotations
                    elif isinstance(e, pyast.BinOp):
                        annotations = self.convert_annotation(e.right)
                        self.type_annotations[Name(e.left)] = annotations
                    else:
                        parse_requires(False, "Unknown expression: " + astor.dump(e))
                elif isinstance(e, pyast.ImportFrom) or isinstance(e, pyast.Import):
                    parse_requires(isinstance(e, pyast.Import), "Import from not supported.")
                    self.imports.update({Name(n.name): n.asname for n in e.names})
                else:
                    parse_expects(False, astor.dump(e))


def parse(filename, include_path, extra_args):
    try:
        ast = astor.parse_file(filename)
    except SyntaxError as e:
        # Catch and convert syntax error ASAP to avoid hiding real syntax errors
        parse_requires(False, e, loc=Location(e.filename, e.lineno, None, None))
    # print(astor.to_source(ast))
    # print(astor.dump(ast))

    return Converter().convert(ast, filename)

    # spec = importlib.util.spec_from_file_location("__nw_config__", filename)
    # module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(module)
    # for n, v in module.__dict__.items():
    #     if n.startswith("__") and n.endswith("__"):
    #         continue
    #     print(n, v)
