#!/usr/bin/python3

import argparse
import hashlib
import logging
import os
import pathlib
import sys
import tarfile
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--specs", nargs="+", default=[])
    args = parser.parse_args()

    download_llvm_lib()
