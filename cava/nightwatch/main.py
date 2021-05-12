import argparse
import logging
import pathlib
import sys

from . import *

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="NightWatch generator")
    parser.add_argument("inputfile", metavar="FILENAME", type=str, help="The NightWatch file to parse.")
    parser.add_argument("--language", "-x", type=str, default=None, help="The language of the API being parsed.")
    parser.add_argument(
        "-I", type=str, action="append", dest="include_path", help="Add a path to the include search path."
    )
    parser.add_argument("-D", type=str, action="append", dest="definitions", help="Define a macro for preprocessing.")
    parser.add_argument("-X", type=str, action="append", dest="extra_args", help="Pass an argument to clang.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Output verbose information (also passed to underlying libraries).",
    )
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        dest="build",
        help="Build the generated code using the generated makefile.",
    )
    parser.add_argument(
        "-u",
        "--missing",
        action="store_true",
        dest="missing",
        help="Output inferred definitions for all functions which appear in the headers, but not "
        "in the NightWatch file. These can be pasted into the NightWatch file and modified to "
        "suit.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Output the API model in roughly the input format. This will loose information.",
    )

    args = parser.parse_args()

    if not args.language:
        if args.inputfile.endswith(".py"):
            args.language = "py"
        elif args.inputfile.endswith(".c"):
            args.language = "c"
        else:
            args.language = "c"

    try:
        errors = []
        if args.language.lower() == "c":
            errors = []

            from pathlib import Path
            here = Path(__file__)
            default_include_path = [
                str(p) for p in [
                    (here / ".." / ".." / "..").resolve(),
                    (here / ".." / ".." / ".." / "include").resolve(),
                    (here / ".." / ".." / ".." / "third_party" / "plog" / "include" ).resolve(),
                ]
            ]

            from .parser import c

            api = c.parse(
                args.inputfile,
                include_path=default_include_path if args.include_path is None else args.include_path + default_include_path,
                definitions=args.definitions or [],
                extra_args=(["-v"] if args.verbose else []) + (args.extra_args or []),
            )

            if args.dump:
                print(api)
            if args.missing:
                from nightwatch.indent import indent_c

                print(indent_c("\n".join(str(f) for f in api.missing_functions)))
            else:
                function_str = ", ".join(str(f.name) for f in api.missing_functions)
                from nightwatch.parser import parse_expects
                from nightwatch.model import Location

                parse_expects(
                    not api.missing_functions,
                    f"""
Functions are missing from {args.inputfile}, but appear in {", ".join(api.includes)}:
{function_str}
(Use -u/--missing to output their inferred specifications.)""".strip(),
                    loc=Location(args.inputfile, None, None, None),
                    kind=info,
                )

            filename_prefix = api.directory_spelling + "/"
            pathlib.Path(api.directory_spelling).mkdir(parents=True, exist_ok=True)

            from .indent import write_file_c
            from .generator import header

            write_file_c(*header.header(api, errors), filename_prefix=filename_prefix)
            write_file_c(*header.utilities_header(api, errors), indent=False, filename_prefix=filename_prefix)
            write_file_c(*header.utility_types_header(api, errors), indent=False, filename_prefix=filename_prefix)
            write_file_c(*header.types_header(api, errors), indent=False, filename_prefix=filename_prefix)
            from .generator.c import guestlib

            write_file_c(*guestlib.source(api, errors), filename_prefix=filename_prefix)

            from .generator.c import worker

            write_file_c(*worker.source(api, errors), filename_prefix=filename_prefix)

            from .generator.c import cmakelists

            write_file_c(*cmakelists.source(api, errors), indent=False, filename_prefix=filename_prefix)

            # TODO(yuhc): Fix build.
            if args.build:
                import subprocess

                try:
                    subprocess.run(
                        ["make", "-C", str(pathlib.Path(api.directory_spelling).resolve()), "clean"], check=True
                    )
                    subprocess.run(
                        ["make", "-C", str(pathlib.Path(api.directory_spelling).resolve()), "all"], check=True
                    )
                except subprocess.CalledProcessError as e:
                    errors.append(
                        LocatedError(
                            error, f"Build returned non-zero exit code {e.returncode}", loc="make", phase="build"
                        )
                    )
        elif args.language.lower().startswith("py"):
            from .parser import python

            api = python.parse(args.inputfile, include_path=args.include_path or [], extra_args=args.extra_args or [])

            if args.dump:
                print(api)

            filename_prefix = api.directory_spelling + "/"

            from .indent import write_file_py
            from .generator.python import guestlib

            write_file_py(*guestlib.source(api, errors))

        if errors:
            raise MultipleError(*errors)
    except (NightWatchError, MultipleError) as e:
        if hasattr(e, "report"):
            e.report()
        print("NightWatch processing failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
