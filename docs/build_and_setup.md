Build and Setup
===============

## Dependencies

A number of packages are required to build a minimal AvA development and
test environment. To use full components and benchmarks, one should follow
the instructions per specification or benchmark to setup the machine.

The following script will check whether Ubuntu 18.04 is running and install
the necessary dependencies.

```shell
sudo ./tools/install_dependencies.sh
```

The following instructions are tested on Ubuntu 18.04 (Linux 4.15) with
GCC 7.5.0, Python 3.6.9, Boost 1.65.x, cmake 3.19.1 and Protobuf 3.0-3.9.

## Configuration

Clone AvA's base repository and initiailize it:

```shell
git clone https://github.com/utcs-scea/ava.git
cd ava
git submodule update --init --recursive
cd ..
mkdir build
cd build
cmake ../ava
```

The above commands initialize a cmake build directory with all flags turned
`OFF`. AvA supports a handful of accelerators and APIs, and the merge of the
old codes into the new build system (v2.0) is still ongoing.

| API framework    | Version | Status   |
| ---------------- | ------- | -------- |
| AmorphOS FPGA    | 1.0     | TESTED   |
| CUDA driver      | 10.1    | TESTED   |
| CUDA runtime     | 10.1    | TESTED   |
| Demo             | 0.0.1   | TESTED   |
| GTI              | 4.5.0.3 | TESTED   |
| HIP              |         | NO       |
| NCSDK v2         | 2.10.01 | TESTED   |
| ONNXruntime CUDA | 10.1    | TESTED\* |
| OpenCL           | 1.2     | TESTED   |
| QuickAssist      | 1.7     | TESTED   |
| TensorFlow CUDA  | 10.1    | TESTED   |
| TensorFlow C     | 2.3.1   | TESTED   |
| Test             | 0.1     | TESTED   |

> \* Upstream changes ([ava-serverless](https://github.com/photoszzt/ava-serverless)) have not been merged.

| AvA manager | Status |
| ----------- | ------ |
| Demo        | TESTED |
| Galvanic    | NO |
| Katana      | NO |
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

> One can run this without using ccmake to configure the build again:
>
> ```shell
> cmake ../ava -DAVA_GEN_DEMO_SPEC=ON -DAVA_MANAGER_DEMO=ON
> ```

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
pushd .
cd ../ava/cava/samples/demo/test_program/
make
popd
```

Start the demo manager by

```shell
./install/bin/demo_manager install/demo_nw/bin/worker
```

Add AvA configuration file:
```
sudo mkdir -p /etc/ava
sudo tee /etc/ava/guest.conf <<EOF
channel = "TCP";
manager_address = "0.0.0.0:3333";
gpu_memory = [1024L];
EOF
```

Run the test program by

```shell
LD_LIBRARY_PATH=install/demo_nw/lib ../ava/cava/samples/demo/test_program/test
```

The demo spec implements and annotates an `int ava_test_api(int x)` API which
returns `x+1` to the caller. In most cases, the API is defined in a separate
shared library and the library is linked to the generated API server. But for
the sake of simplicity, `ava_test_api` is just defined in the spec.

The demo program simply calls an `int ava_test_api(int)` API which is forwarded
to the API server. The API server prints a message `RECEIVED AVA_TEST_API(9999)`
and sends a response to the demo program. The demo program then prints a message
`RECEIVED AVA_TEST_API = 10000`.
