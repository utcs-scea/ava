Build and Setup
===============

## Dependencies

The following packages are required to build a minimal AvA development and
test environment. To use full components and benchmarks, one should follow
the instructions per specification or benchmark to setup the machine.

```shell
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
sudo apt purge --auto-remove cmake
sudo apt install cmake cmake-curses-gui
sudo apt install git build-essential python3 python3-pip libglib2.0-dev clang-7 libclang-7-dev indent libssl-dev
python3 -m pip install pip
python3 -m pip install conan toposort astor 'numpy==1.15.0'
```

The following instructions are tested on Ubuntu 18.04 (Linux 4.15) with
GCC 7.5.0, Python 3.6.9 and cmake 3.19.1.

## Configuration

Clone AvA's base repository and initiailize it:

```shell
git clone https://github.com/utcs-scea/ava.git
cd ava
git submodule update --init --recursive
cd ..
mkdir build
cd build
conan install -if . ../ava/config --build=missing
cmake ../ava
```

The above commands initialize a cmake build directory with all flags turned
`OFF`. AvA supports a handful of accelerators and APIs, and the merge of the
old codes into the new build system (v2.0) is still ongoing.

| API framework    | Status |
| ---------------- | ------ |
| AmorphOS FPGA    | UNTESTED |
| CUDA driver 10   | UNTESTED |
| CUDA runtime 10  | UNTESTED |
| Demo             | NO |
| GTI              | NO |
| HIP              | NO |
| NCSDK v2         | UNTESTED |
| ONNXruntime CUDA | UNTESTED |
| OpenCL           | NO |
| QuickAssist      | NO |
| TensorFlow CUDA  | UNTESTED |
| TensorFlow C     | NO |
| Test             | NO |

| AvA manager | Status |
| ----------- | ------ |
| Demo        | NO |
| Galvanic    | UNTESTED |
| Katana      | UNTESTED |
| Nvidia GPU  | NO |

This tutorial shows how to configure and build the minimal AvA demo API
remoting system. To build and use other supported virtualized APIs, please
reference respective documentations. To virtualize a new API with AvA, please
reference the [virtualize_new_api](virtualize_new_api.md) tutorial.

In the `build` directory, run:

```shell
ccmake .
```

Then turn on `AVA_GEN_DEMO_SPEC` and `AVA_MANAGER_DEMO` and press `c` to
reconfigure the build.

## Build and Run

Build the demo system simply by:

```shell
make -j`nproc`
```

This generates the guestlib and API server codes in `ava/cava/demo_nw` and
builds their binary files to `build/install/demo_nw`. The demo manager is
compiled and installed into `build/install/bin`.

```shell
ls install/demo_nw/bin
ls install/demo_nw/lib
ls install/bin
```

AvA's base repository includes a tiny standalone demo program.

```shell
TODO: steps to build the test program.
```

```shell
TODO: steps to run worker (demo manager) and run test program.
```

The demo spec implements and annotates an `int ava_test_api(int x)` API which
returns `x+1` to the caller. In most cases, the API is defined in a separate
shared library and the library is linked to the generated API server. But for
the sake of simplicity, `ava_test_api` is just defined in the spec.

The demo program simply calls an `int ava_test_api(int)` API which is forwarded
to the API server. The API server prints a message `RECEIVED AVA_TEST_API(9999)`
and sends a response to the demo program. The demo program then prints a message
`RECEIVED AVA_TEST_API = 10000`.
