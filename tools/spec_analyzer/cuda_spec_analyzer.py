"""
CUDA spec analyzer
Dumps CSV spreadsheet summary to analysis.csv
Dumps combined spec information function content to analysis.txt
"""

#!/usr/bin/env python3

from typing import List, Optional, Tuple
import argparse
import csv
import re
from pathlib import Path
from glob import glob


PWD = (Path(__file__) / "..").resolve()
AVA_SPECS_PATH = (PWD / ".." / ".." / "cava" / "samples").resolve().as_posix()
OUTPUT_FILENAME = "analysis"


def parse_function_info(filename: str) -> Optional[List[Tuple[str, str, str]]]:
    c_src = open(filename).read()

    # Skip spec files which aren't CUDA related (look for cuda_runtime(_api).h)
    if "cuda_runtime" not in c_src:
        return None

    # Remove multi-line comments
    c_src = re.sub(r"/\*.*?\*/", "", c_src, flags=re.DOTALL)
    # Remove single line comments
    c_src = re.sub(r"//.*?\n", "\n", c_src)
    # Remove __dv(VAL) from function arguments
    c_src = re.sub(r" +__dv *\([^)]+?\) *", "", c_src)
    # Remove preprocessor commands
    c_src = re.sub(r"\n#[^\n]+", "", c_src)

    # Find functions and return a tuple for each function which has:
    #   function preamble - content up to and including name,
    #   arguments
    #   function body
    return re.findall(r"\n\n([^\s][^(]+)(\(.*?\))(?:;|\s*{(.*?)\n})", c_src, flags=re.M | re.DOTALL)


# pylint: disable=too-many-branches,too-many-statements
def analyze_specs(output_path: Path):
    # Dict of function names to info dict
    # { 'preamble' : str,
    #   'args' : str,
    #   'sigs' : {
    #      SIG : {'index' : int, 'body' : str, 'specs' : set ([1, 2, ..])}
    #    }
    # }
    func_info = {}
    spec_idx = 0
    cuda_spec_names = []  # Array of spec names (base name of cpp file)

    filenames = sorted(
        glob(AVA_SPECS_PATH + "/**/*.cpp", recursive=True), key=lambda x: re.sub("[^/]+/", "", x).replace(".cpp", "")
    )
    for src_file in filenames:
        finfo = parse_function_info(src_file)
        if not finfo:
            continue

        # Add to function info dictionary
        for preamble, arguments, body in finfo:
            # Parse out function names from function preamble
            name = re.split(r"\s", preamble.strip())[-1]
            if not name.startswith("cu"):
                continue

            # Create or fetch function info by name
            if name not in func_info:
                info = {"preamble": preamble, "args": arguments, "sigs": {}}
                func_info[name] = info
            else:
                info = func_info[name]

            # Is function implemented?
            if "%s is not implemented" not in body:
                # Use the function body content without whitespace as its "signature"
                sig = re.sub(r"\s", "", body)
            else:
                # Use empty signature for unimplemented
                sig = ""

            # Add new body signature if unique, or add spec to the list which implements the
            # same function content
            if sig not in info["sigs"]:
                info["sigs"][sig] = {"index": len(info["sigs"]), "body": body, "specs": set([spec_idx])}
            else:
                info["sigs"][sig]["specs"].update([spec_idx])

        cuda_spec_names.append(re.sub("[^/]+/", "", src_file).replace(".cpp", ""))
        spec_idx += 1

    # Use specification base names for CSV columns
    csv_rows = [["Function"] + cuda_spec_names]
    counts = [0] * len(cuda_spec_names)
    func_names = sorted(func_info.keys())

    # Construct CSV rows
    for name in func_names:
        row = [name]
        sig_dict = func_info[name]["sigs"]

        for i in range(len(cuda_spec_names)):
            for sig, sig_info in sig_dict.items():
                if i in sig_info["specs"]:
                    if sig != "":
                        row.append(str(sig_info["index"]))
                        counts[i] += 1
                    else:
                        row.append("STUB")
                    break
            else:
                row.append("")

        csv_rows.append(row)

    csv_rows.append(["Totals:"] + counts)
    csv_output_file = (output_path / (OUTPUT_FILENAME + ".csv")).resolve().as_posix()
    csv.writer(open(csv_output_file, "w")).writerows(csv_rows)

    # Write out specs function overview document
    txt = ""
    for name in func_names:
        info = func_info[name]
        txt += "// " + 80 * "=" + "\n"
        txt += info["preamble"] + info["args"] + "\n"

        not_present = set(range(len(cuda_spec_names)))
        for sig_info in info["sigs"].values():
            not_present -= sig_info["specs"]

        if not_present:
            txt += "// UNAVAILABLE: " + ",".join([cuda_spec_names[i] for i in sorted(list(not_present))])

        if "" in info["sigs"]:
            txt += "// UNIMPLEMENTED: " + ",".join(
                [cuda_spec_names[i] for i in sorted(list(info["sigs"][""]["specs"]))]
            )

        first = True
        for sig, sig_info in info["sigs"].items():
            if sig == "":
                continue

            if not first:
                first = False
            else:
                txt += "// " + 20 * "-" + "\n"
            txt += "// Implemented: " + ",".join([cuda_spec_names[i] for i in sorted(list(sig_info["specs"]))])
            txt += sig_info["body"] + "\n"

    txt_output_file = (output_path / (OUTPUT_FILENAME + ".txt")).resolve().as_posix()
    open(txt_output_file, "w").write(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=PWD.as_posix(), help="Output directory")
    args = parser.parse_args()

    analyze_specs(Path(args.output))
