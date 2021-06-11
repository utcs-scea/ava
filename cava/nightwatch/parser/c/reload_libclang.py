from pathlib import Path
import sys
from glob import glob


clang_version = 7
clang_directories = glob("/usr/lib*/llvm-{v}/lib/clang/{v}*/include".format(v=clang_version)) + glob(
    "/usr/lib*/clang/{v}*/include".format(v=clang_version)
)
clang_flags = (
    [
        "-Wno-return-type",
        "-Wno-empty-body",
        "-nobuiltininc",
    ]
    + [a for d in clang_directories for a in ["-isystem", d]]
    + ["-I../worker/include"]
)


def _config_clang() -> None:
    here = Path(__file__)
    llvm_build_python = (here / ".." / ".." / ".." / ".." / ".." / "llvm" / "clang" / "bindings" / "python").resolve()
    if llvm_build_python.exists():
        sys.path.insert(0, str(llvm_build_python))
    # pylint: disable=import-outside-toplevel
    from clang import cindex

    llvm_build_lib = (here / ".." / ".." / ".." / ".." / ".." / "llvm" / "build" / "lib").resolve()
    clang_so_names = [s.format(clang_version) for s in ["libclang.so.{}", "libclang-{}.so"]]
    for name in [str(llvm_build_lib / s) for s in clang_so_names] + clang_so_names:
        cindex.Config.set_library_file(name)
        try:
            cindex.conf.get_cindex_library()
            break
        except cindex.LibclangError:
            continue


_config_clang()
