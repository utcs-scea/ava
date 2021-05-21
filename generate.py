#!/usr/bin/python3

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
GLIB2_CFLAGS = pkgconfig.cflags("glib-2.0").split(" ")

SPEC_LIST = {
    "cudadrv": ("samples/cudadrv/cuda_driver.c", ["-I/usr/local/cuda-10.1/include"]),
    "cudart": ("samples/cudart/cudart.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
    "tf_dump": ("samples/tensorflow/tf_dump.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
    "tf_opt": ("samples/tensorflow/tf_opt.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
    "onnx_dump": ("samples/tensorflow/onnx_dump.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
    "onnx_opt": ("samples/tensorflow/onnx_opt.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
    "pt_dump": ("samples/tensorflow/pt_dump.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
    "pt_opt": ("samples/tensorflow/pt_opt.cpp", ["-I/usr/local/cuda-10.1/include", "-Iheaders"] + GLIB2_CFLAGS),
}


def generate_code(spec_name: str):
    if spec_name not in SPEC_LIST:
        logger.warning(f"Unsupported {spec_name} specification")
        return

    spec_file, spec_parameter = SPEC_LIST[spec_name]
    print(spec_file)
    print(["./nwcc", spec_file] + spec_parameter)
    _ = subprocess.run(["./nwcc", spec_file] + spec_parameter, cwd=CAVA_DIR, check=True)
    logger.info(f"Code generation for {spec_name} specification is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--specs", nargs="+", default=[])
    args = parser.parse_args()

    download_llvm_lib()

    for spec in args.specs:
        generate_code(spec)
