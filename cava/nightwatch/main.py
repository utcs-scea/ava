import argparse
import logging
import pathlib
import sys
from pathlib import Path
from typing import List

from nightwatch import info, MultipleError, NightWatchError
from nightwatch.indent import indent_c, write_file_c
from nightwatch.parser import c, parse_expects
from nightwatch.model import Location, API
from nightwatch.generator import header
from nightwatch.generator.c import guestlib, worker, cmakelists


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_c(
    inputfile: str,
    include_path: List[str],
    definitions: List[str],
    extra_args: List[str],
    verbose: bool = False,
    print_missing: bool = False,
) -> API:
    here = Path(__file__)
    default_include_path = [
        str(p)
        for p in [
            (here / ".." / ".." / "..").resolve(),
            (here / ".." / ".." / ".." / "include").resolve(),
            (here / ".." / ".." / ".." / "third_party" / "plog" / "include").resolve(),
        ]
    ]

    try:
        api = c.parse(
            inputfile,
            include_path=default_include_path if include_path is None else include_path + default_include_path,
            definitions=definitions or [],
            extra_args=(["-v"] if verbose else []) + (extra_args or []),
        )

        if print_missing:
            print(indent_c("\n".join(str(f) for f in api.missing_functions)))
        else:
            function_str = ", ".join(str(f.name) for f in api.missing_functions)
            parse_expects(
                not api.missing_functions,
                f"""
Functions are missing from {inputfile}, but appear in {", ".join(api.includes)}:
{function_str}
(Use -u/--missing to output their inferred specifications.)""".strip(),
                loc=Location(inputfile, None, None, None),
                kind=info,
            )

    except (NightWatchError, MultipleError) as e:
        if hasattr(e, "report"):
            e.report()
        print("NightWatch processing failed.", file=sys.stderr)
        sys.exit(1)

    return api


def generate_c(api: API):
    try:
        errors = []
        filename_prefix = api.directory_spelling + "/"
        pathlib.Path(api.directory_spelling).mkdir(parents=True, exist_ok=True)

        write_file_c(*header.header(api, errors), filename_prefix=filename_prefix)
        write_file_c(*header.utilities_header(api), indent=False, filename_prefix=filename_prefix)
        write_file_c(*header.utility_types_header(api), indent=False, filename_prefix=filename_prefix)
        write_file_c(*header.types_header(api), indent=False, filename_prefix=filename_prefix)
        write_file_c(*guestlib.source(api), filename_prefix=filename_prefix)
        write_file_c(*worker.source(api), filename_prefix=filename_prefix)
        write_file_c(*cmakelists.source(api), indent=False, filename_prefix=filename_prefix)

        if errors:
            raise MultipleError(*errors)
    except (NightWatchError, MultipleError) as e:
        if hasattr(e, "report"):
            e.report()
        print("Cava processing failed.", file=sys.stderr)
        sys.exit(1)


def cava_main():
    parser = argparse.ArgumentParser(description="AvA specification compiler")
    parser.add_argument("inputfile", metavar="FILENAME", type=str, help="The specification file to parse.")
    parser.add_argument("--language", "-x", type=str, default="c", help="The language of the API being parsed.")
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
        "-u",
        "--missing",
        action="store_true",
        dest="missing",
        help="Output inferred definitions for all functions which appear in the headers, but not "
        "in the specification file. These can be pasted into the specification file and modified "
        "to suit.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Output the API model in roughly the input format. This will loose information.",
    )
    parser.add_argument(
        "--optimization",
        "-O",
        type=str,
        action="append",
        dest="optimizations",
        choices=["batching"],
        help="Enable code generation for AvA optimizations.",
    )
    args = parser.parse_args()

    if args.language.lower() == "c":
        assert args.inputfile.endswith(".c") or args.inputfile.endswith(
            ".cpp"
        ), "Expected input file extension to be c or cpp."
    else:
        raise RuntimeError(f"Unsupported language: {args.language}.")

    if args.language.lower() == "c":
        api = parse_c(
            args.inputfile,
            args.include_path,
            args.definitions,
            args.extra_args,
            verbose=args.verbose,
            print_missing=args.missing,
        )
        api.enable_optimizations(args.optimizations)
        if args.dump:
            print(api)
        generate_c(api)


if __name__ == "__main__":
    cava_main()
