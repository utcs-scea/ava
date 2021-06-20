#!/usr/bin/env python3

from typing import List

import argparse
import hashlib
import logging
import os
import pathlib
import subprocess
import sys
import tarfile

import pkgconfig
import wget

LLVM_DIR = os.path.dirname(os.path.realpath(sys.argv[0])) + "/llvm"
CLANG_LIB_DIR = LLVM_DIR + "/build"
SENTINEL_FILE = ".unpacked"
LLVM_URL = "https://github.com/utcs-scea/ava-llvm/releases/download/v7.1.0/ava-llvm-release-7.1.0.tar.gz"
LLVM_MD5 = "400832522ed255314d6a5848e8f7af6c"


logger = logging.getLogger(__file__)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %I:%M:%S %p")


def download_llvm_lib():
    if not os.path.exists(LLVM_DIR):
        raise RuntimeError("LLVM submodule has not been initialized")
    pathlib.Path(CLANG_LIB_DIR).mkdir(parents=False, exist_ok=True)

    # Download LLVM tarball
    download_llvm = True
    download_target = CLANG_LIB_DIR + "/" + LLVM_URL.split("/")[-1]
    if os.path.isfile(download_target):
        if LLVM_MD5 == hashlib.md5(open(download_target, "rb").read()).hexdigest():
            download_llvm = False
    if download_llvm:
        wget.download(LLVM_URL, download_target)
        print()
        logger.info(f"ava-llvm downloaded to {download_target}")
    else:
        logger.info(f"ava-llvm ready exists at {download_target}")

    # Unzip downloaded LLVM tarball
    sentinel_target = CLANG_LIB_DIR + "/" + SENTINEL_FILE
    unzip_llvm = not os.path.isfile(sentinel_target)
    if download_llvm or unzip_llvm:
        with tarfile.open(download_target) as f:
            f.extractall(LLVM_DIR)
        logger.info(f"ava-llvm unpacked to {CLANG_LIB_DIR}")
        with open(sentinel_target, "w") as f:
            logger.info(f"Sentinel file created at {sentinel_target}")
    else:
        logger.info(f"ava-llvm already unpacked to {CLANG_LIB_DIR}")


CAVA_DIR = os.path.dirname(os.path.realpath(sys.argv[0])) + "/cava"
CUDA_10_1_CFLAGS = "-I/usr/local/cuda-10.1/include -I/usr/local/cuda-10.1/nvvm/include".split(" ")
GLIB2_CFLAGS = pkgconfig.cflags("glib-2.0").split(" ")
FMT_CFLAGS = ["-I" + os.path.dirname(os.path.realpath(sys.argv[0])) + "/third_party/fmt/include"]


def check_cflags(force_build: bool = False):
    any_warning = False

    def include_dir_exists(cflag: str) -> bool:
        assert cflag.startswith("-I"), "Invalid header include flag"
        if not os.path.exists(cflag[2:]):
            logger.warning(f"Include directory not found: {cflag[2:]}")
            return False
        return True

    if not GLIB2_CFLAGS:
        logger.warning("GLIB2_CFLAGS is empty. Are you running in a virtual environment?")
        any_warning = True

    for cflag in CUDA_10_1_CFLAGS + GLIB2_CFLAGS:
        if cflag.startswith("-I"):
            if not include_dir_exists(cflag):
                any_warning = True

    if any_warning and not force_build:
        input("Press Enter to continue...")


SPEC_LIST = {
    "cudadrv": ("samples/cudadrv/cuda_driver.c", [] + CUDA_10_1_CFLAGS),
    "cudart": ("samples/cudart/cudart.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
    "demo": ("samples/demo/demo.c", ["-Iheaders"]),
    "gti": ("samples/gti/gti.c", []),
    "ncsdk": ("samples/ncsdk/mvnc.c", []),
    "onnx_dump": ("samples/onnxruntime/onnx_dump.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
    "onnx_opt": ("samples/onnxruntime/onnx_opt.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
    "opencl": ("samples/opencl/opencl.c", []),
    "pt_dump": ("samples/pytorch/pt_dump.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
    "pt_opt": ("samples/pytorch/pt_opt.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
    "qat": (
        "samples/quickassist/qat.c",
        [
            f"-I{os.getenv('ICP_ROOT')}/quickassist/include",
            f"-I{os.getenv('ICP_ROOT')}/quickassist/lookaside/access_layer/include",
        ],
    ),
    "test": ("samples/test/libtrivial.c", ["-I../test"]),
    "tf_c": ("samples/tensorflow_c/tf_c.c", []),
    "tf_dump": ("samples/tensorflow/tf_dump.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
    "tf_opt": ("samples/tensorflow/tf_opt.cpp", ["-Iheaders"] + CUDA_10_1_CFLAGS + GLIB2_CFLAGS + FMT_CFLAGS),
}


def generate_code(spec_name: str, enabled_optimizations: List[str] = None):
    if spec_name not in SPEC_LIST:
        logger.warning(f"Unsupported {spec_name} specification")
        return

    spec_file, spec_parameter = SPEC_LIST[spec_name]
    opt_parameter = []
    if enabled_optimizations and len(enabled_optimizations) > 0:
        opt_parameter = ["--optimization"] + enabled_optimizations

    _ = subprocess.run(["./nwcc", spec_file] + opt_parameter + spec_parameter, cwd=CAVA_DIR, check=True)
    logger.info(f"Code generation for {spec_name} specification is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--specs", nargs="+", default=[], choices=SPEC_LIST.keys(), help="Specification shortnames"
    )
    parser.add_argument("-f", "--force", action="store_true", help="Build specifications regardless any warnings")
    parser.add_argument(
        "-O",
        "--opt",
        type=str,
        action="append",
        dest="optimizations",
        choices=["batching"],
        help="Enable optimizations",
    )
    args = parser.parse_args()

    download_llvm_lib()

    check_cflags(args.force)

    for spec in args.specs:
        generate_code(spec, args.optimizations)
