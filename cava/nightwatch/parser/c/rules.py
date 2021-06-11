from typing import Union

# pylint: disable=unused-import
import nightwatch.parser.c.reload_libclang
from clang.cindex import Cursor, CursorKind, Type, TypeKind
from clang import cindex
from nightwatch import model
from nightwatch.annotation_set import AnnotationSet


class Rule:
    def __init__(self, annotations: AnnotationSet) -> None:
        self.annotations = annotations

    def matches(self, ct, data):
        """
        Return true if the Cursor or Type is matched by this rule.
        """
        raise NotImplementedError()

    def apply(self, ct: Union[Cursor, Type], data: AnnotationSet) -> None:
        assert hasattr(ct, "kind")
        if self.matches(ct, data):
            # print(f"Rule {self} matched {ct.spelling}: adding {self.annotations} to {data}")
            for k, v in self.annotations.items():
                data.setdefault(k, v)
        else:
            # print(f"Rule {self} did not match {ct.spelling}")
            pass

    def __str__(self):
        return f"{type(self).__name__}{self.__dict__}"


class CursorRule(Rule):
    def matches(self, ct, data):
        return isinstance(ct, Cursor)


class TypeRule(Rule):
    def matches(self, ct: Union[Cursor, Type], data: AnnotationSet) -> bool:
        return (
            isinstance(ct, cindex.Type)
            and self._matches_type(ct, data)
            or isinstance(ct, cindex.Cursor)
            and ct.kind != CursorKind.FUNCTION_DECL
            and self._matches_type(ct.type, data)
        )

    def _matches_type(self, ct, data):
        raise NotImplementedError()


class Functions(Rule):
    def matches(self, ct, data):
        return ct.kind == CursorKind.FUNCTION_DECL


class PointerTypes(TypeRule):
    def _matches_type(self, ct: Type, data: AnnotationSet) -> bool:
        return ct.is_data_pointer() and "transfer" not in data


class ConstPointerTypes(PointerTypes):
    def _matches_type(self, ct: Type, data: AnnotationSet) -> bool:
        return super()._matches_type(ct, data) and (ct.is_const_qualified() or ct.get_pointee().is_const_qualified())


class NonconstPointerTypes(PointerTypes):
    def _matches_type(self, ct: Type, data: AnnotationSet) -> bool:
        return super()._matches_type(ct, data) and not (
            ct.is_const_qualified() or ct.get_pointee().is_const_qualified()
        )


class Types(TypeRule):
    def __init__(self, tpe: Type, annotations: AnnotationSet) -> None:
        super().__init__(annotations)
        self.type = tpe
        # pylint: disable=protected-access
        self.type_str = model.Type._drop_const(self.type.spelling)

    def _is_correct_type(self, ct: Type) -> bool:
        # TODO: This should compare the types instead of their spellings, but type comparison is returning false.
        # pylint: disable=protected-access
        return model.Type._drop_const(ct.spelling) == self.type_str

    def _matches_type(self, ct: Type, data: AnnotationSet) -> bool:
        return self._is_correct_type(ct)


class NonTransferableTypes(TypeRule):
    def _matches_type(self, ct: Type, data: AnnotationSet) -> bool:
        """
        Walk down through pointers looking for a type that is incomplete (has no size).
        """

        def nontransferrable(ct, internal):
            nontransferrable_type = (
                ct.get_size() < 0 and ct.kind != TypeKind.VOID and ct.kind != TypeKind.INCOMPLETEARRAY
            )
            nontransferrable_struct = any(
                f.type.is_pointer() or nontransferrable(f.type, True) for f in ct.get_fields()
            )
            pointer_to_nontransferrable = not internal and ct.is_pointer() and nontransferrable(ct.get_pointee(), True)
            return nontransferrable_type or nontransferrable_struct or pointer_to_nontransferrable

        return nontransferrable(ct.get_canonical(), False)


# class ComputeSizes(Rule):
#     def __init__(self):
#         super().__init__({})
#
#     def matches(self, ct, data):
#         return isinstance(ct, cindex.Type) or ct.type is not None
#
#     def apply(self, ct, data):
#         if self.matches(ct, data):
#             if ct.kind == CursorKind.FUNCTION_DECL:
#                 type = ct.result_type
#                 name = RET_ARGUMENT_NAME
#             elif isinstance(ct, Cursor):
#                 type = ct.type
#                 name = data["name"] or ct.displayname
#             else:
#                 type = ct
#                 name = data.get("name", None)
#
#             # data["direct_size"] = f"sizeof({type.spelling})"
#             # print(name, ct.spelling, data)
#
#             if type.is_data_pointer() and "handle" not in data and "opaque" not in data:
#                 data.setdefault("buffer", "1")
