Automatic Virtualization of Accelerators
========================================

AvA is a research system for virtualizing general API-controlled
accelerators automatically, developed in SCEA lab at University of Texas
at Austin. AvA is prototyped on KVM and QEMU, compromising compatibility
with automation for classic API remoting systems, and introducing
hypervisor interposition for resource management and strong isolation.

This repository is the main codebase of AvA. We host customized Linux
kernel, QEMU, LLVM, and sets of benchmarks in separate repositories.

Related repositories
--------------------

* Kernel: https://github.com/utcs-scea/ava-kvm
* QEMU: https://github.com/utcs-scea/ava-qemu
* LLVM: https://github.com/utcs-scea/ava-llvm
* Benchmarks: https://github.com/utcs-scea/ava-benchmarks

### Setup code tree

Clone the repositories into a single directory:

```
$ git clone git@github.com:utcs-scea/ava.git
$ cd ava
$ git clone https://github.com/utcs-scea/ava-benchmarks benchmark
$ git clone https://github.com/utcs-scea/ava-llvm llvm
$ git clone https://github.com/utcs-scea/ava-qemu qemu
$ git clone https://github.com/utcs-scea/ava-kvm  kvm
```

Install
-------

### Dependencies

The following packages are required to build and run all AvA
components and benchmarks:

* Ubuntu (apt install): git libssl-dev libglib2.0-dev
  libpixman-1-dev opencl-headers curl bc build-essential 
  libclang-7-dev clang-7 ctags caffe-cpu pssh python3
  python3-pip virtualenv indent
* Python3 (pip3 install): blessings toposort astor numpy(==1.15.0)

AvA was fully tested on Ubuntu 18.04 with GCC 5.5+, Python 3.6+,
Bazel 0.26.1, and customized LLVM 7.0, Linux kernel 4.14, and QEMU 3.1.
The system also works in Ubuntu 16.04 with additional care of Python 3.6
installation for CAvA scripts.

### Hardware and API

The following hardware and APIs are virtualized with AvA (excluding
manually implemented Python forwarding):

| API framework           | Hardware                     |
|-------------------------|------------------------------|
| OpenCL 1.2              | NVIDIA GTX 1080 / AMD RX 580 |
| CUDA 10.0 (driver)      | NVIDIA GTX 1080              |
| CUDA 10.0 (runtime)     | NVIDIA GTX 1080              |
| TensorFlow 1.12 C       | Intel Xeon E5-2643           |
| TensorFlow 1.14 Python  | NVIDIA GTX 1080              |
| NCSDK v2                | Intel Movidius NCS v1 & v2   |
| GTI SDK 4.4.0.3         | Gyrfalcon 2803 Plai Plug     |
| QuickAssist 1.7         | Intel QuickAssist            |
| Custom FPGA on AmorphOS | AWS F1                       |

### Environment

Export the following variables in bash profile:

| Name             | Example               | Explanation                             |
|------------------|-----------------------|-----------------------------------------|
| AVA_ROOT         | /project/ava          | Path to AvA source tree                 |
| AVA_CHANNEL      | SHM                   | Transport channel (SHM\|TCP\|VSOCK\|LOCAL) |
| AVA_MANAGER_ADDR | 0.0.0.0:3333          | (guestlib only) AvA manager's address   |
| AVA_WPOOL        | TRUE                  | Enable API server pool                  |
| DATA_DIR         | /project/rodinia/data | Path to Rodinia dataset                 |

More environment variables are introduces in CAvA and benchmarks'
documentations.

### Install Kernel (KVM)

The host kernel needs to be replaced to enable VSOCK interposition and
resource management (`LOCAL` channel can run with native kernel without
interposition). We provide kernel configuration and Kbuild script in
`$AVA_ROOT/kbuild`. `KVM` and `VHOST_VSOCK` modules are required.

```
## In $AVA_ROOT/kbuild
$ cp 4.14.config .config && make -f Makefile.setup
$ make -j16 && make modules
$ sudo make modules_install ; sudo make install 
```

Reboot and verify the installed kernel by `uname -r`. Because AvA depends
on `VHOST_VSOCK` module, it needs to be loaded by `sudo modprobe vhost_vsock`
before starting any AvA components. Or it can be automatically loaded by

```
## In /etc/modules-load.d/modules.conf
vhost_vsock
## Reboot the machine
$ sudo reboot
```

## Compile QEMU

Compile QEMU in the normal way as below. The VM boot script will point
to `$AVA_ROOT/qemu` directory, so there is no need to install the compiled
binary.

```
## In $AVA_ROOT/qemu
$ ./configure
$ make -j16
```

## Compile LLVM

Build the artifacts into `$AVA_ROOT/llvm/build` which are required by CAvA.
You do not need to install them. CAvA will find them in their build locations.

```
## In $AVA_ROOT/llvm
$ mkdir build
$ cd build
$ cmake -DLLVM_ENABLE_PROJECTS=clang \
        -G "Unix Makefiles" ../llvm
$ make
```

### Compile generated stack and API server

CAvA is written in Python3. To verify the Python package installation and
CAvA's help menu, please run `./nwcc -v` in `$AVA_ROOT/cava`.

```
## In $AVA_ROOT/cava
$ ./nwcc samples/opencl.nw.c
$ ./nwcc samples/cuda.nw.c \
         -I /usr/local/cuda-10.0/include
$ ./nwcc samples/cudart.nw.c \
         -I headers
         -I /usr/local/cuda-10.0/include \
         `pkg-config --cflags glib-2.0`
$ ./nwcc samples/mvnc.nw.c
```

The guest libraries and workers for those APIs are generated into `cl_nw`,
`cu_nw`, `cudart_nw`, `mvnc_nw`, correspondingly.
They are compiled by `make R=1` for release version, or just `make` with
debugging messages enabled.

### API server manager

Simply execute `make` in `$AVA_ROOT/worker`.

### Support for Ubuntu 16.04

Installing the missing packages will just work fine:

```
$ wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
$ sudo add-apt-repository \"deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main\"
$ sudo add-apt-repository ppa:jonathonf/python-3.6
$ sudo apt update
$ sudo apt install libclang-7-dev
$ sudo apt install python3.6
```

`SOL_NETLINK` socket option is supported in the 16.04 kernel, but it is not
in the libc headers. However libc passes this option though to the kernel so
defining it here should be just as good.

### Zero-copy host driver (optional)

Simply execute `make` in `$AVA_ROOT/hostdrv`.
The zero copy mode is enabled if the compiled driver (zcopy.ko) is installed
before the VM is booted.

Benchmarking
------------

This section will show an example of utilizing virtualized accelerators in VM.

### Start API server manager

Link the compiled API server binary to `$AVA_ROOT/worker` and start manager.

```
## In $AVA_ROOT/worker
$ ln -s $AVA_ROOT/cava/cl_nw/worker worker
$ sudo -E ./manager
```

### Setup virtual machine

The example boot scripts are located in `$AVA_ROOT/vm`.
Update `DIR_QEMU` in `scripts/environment` for compiled QEMU binary,
and `IMAGE_FILE` for the VM disk image.

The IP address is specified by `tapno` in `scripts/options`. By default,
the first VM can be accessed by `ssh localhost -p 2222`.

Before boot the VM by `./run.sh`,
make sure that `sudo -E ./manager` has been started in `$AVA_ROOT/worker`,
and `vhost_vsock` kernel module has been installed (by `sudo modprobe vhost_vsock`).
In the QEMU parameter list, you need to

* Enable KVM: `-enable-kvm -machine accel=kvm -cpu host,kvm=on`
* Assign guest-cid (starts from 3): `-device vhost-vsock-pci,guest-cid=5`
* Enable AvA transport device: `-device ava_vdev`

### Setup guest environment

Once the guest VM is booted, the `guestdrv` source and compiled `guestlib`
need to be copied to VM. Furthermore, the benchmark must load `guestlib`
instead of original framework library at runtime. Run `$AVA_ROOT/vm/setup.sh`
to setup guestdrv and guestlib in guest automatically:

```
## In $AVA_ROOT/vm
## Copy necessary files (e.g., $AVA_ROOT/cava/cl_nw) to
## VM ($GUEST_AVA_ROOT defined in the script), compile
## and install guestdrv
$ ./setup.sh cl_nw
```

Then export `AVA_ROOT` and `DATA_DIR` in guest. Make sure `AVA_CHANNEL` and
`AVA_WPOOL` are consistent between guest and host.

> Note: It is necessary to install essential tool chains to compile benchmarks
> in guest, e.g. CUDA `nvcc`.

In the prototype, because the virtual transport device is built on top of
vhost_vsock, AvA requires the VSOCK version in the guest matches that in the
host. Because AvA host uses 4.14 kernel, the guest VM needs to use kernel 4.10+.

### Compile benchmark and link against libguestlib.so

It is highly recommended to run the benchmark first in the host with native
library to verify the framework installation.

Example for running a single benchmark:

```
## In (guest) $AVA_ROOT/benchmark/rodinia/opencl/backprop
$ make && ./run
```

To run the same benchmark in host, comment \lstinline|make.mk:7-8| before compiling
the benchmark:

```
## In $AVA_ROOT/benchmark/rodinia/opencl/util/make.mk
# OPENCL_LIB = $(AVA_ROOT)/cava/cl_nw
# LIB_NAME = guestlib
```

Minimal remoting system
-----------------------

AvA supports a minimal remoting system without replacing the host driver or
running a VM. The minimal system does not have hypervisor interposition and both
guestlib and worker run in the host.

To enable the minimal system:

* Switch the communication channel to `LOCAL` mode: `export AVA_CHANNEL=LOCAL`.
* Run `sudo -E ./manager_tcp` instead of `./manager` in `$AVA_ROOT/worker`.
* Run the application (compiled against libguestlib.so) in host.

Hard-coded parameters
---------------------

AvA moved a part of configurations to be environment variables, while kept others
which are not commonly used hard-coded.

Most configurations can be found in `$AVA_ROOT/devconf.h`, where the parameter
names and comments imply the meanings. For the demo, you can use the default settings.

To enable AvA system-wide debugging prints, define `DEBUG` in `$AVA_ROOT/include/debug.h`.

AvA requires a clean rebuild after changing the hard-coded configurations.

Environment variables
---------------------

The following variables must be set and consistent in the guest VM and host.

* Communication channel `export AVA_CHANNEL=LOCAL|SHM|VSOCK|TCP`.
  The default value is `LOCAL` when `AVA_CHANNEL` is unset.
  The parameter is read by both the guestlib and the worker to determine the transport method.
  To use LOCAL or TCP channel, the `manager_tcp` (instead of `manager`) is required to be
  started.

* AvA manager host address `export AVA_MANAGER_ADDR=<Server name or IP addreses:port>`.
  The variable must be set for guestlib. If the address is barely a port, the server name
  will use `localhost`.

* Worker pooling `export AVA_WPOOL=TRUE|FALSE`.
  The default is `FALSE` when `AVA_WPOOL` is unset.
  This applies to the manager to pre-initialize a worker pool in the host.

The following variables must be set for special features in only the host.

* Migration random call id `export AVA_MIGRATION_CALL_ID=r<limit>` (such as `r100`) or `<number>`.
  When set to `r<limit>` the runtime will migrate at a random call with number less than `limit`.
  When set to `<number>` the runtime will migrate at exactly call `number`.

Configuring CAvA's Generated Code
---------------------------------

AvA API specification can contain C flags to enable or disable some features during prototyping,
for example, `ava_cflags(-DAVA_RECORD_REPLAY)` enables the record-and-replay feature.

* `AVA_RECORD_REPLAY`: when defined, API record-and-replay is enabled.
* `AVA_API_FUNCTION_CALL_RESOURCE`: when defined, resource reporting annotations take effects.
* `AVA_DISABLE_HANDLE_TRANSLATION`: when defined, the handle id translation is disabled and the
  actual address is used as the handle id.
* `AVA_BENCHMARKING_MIGRATE`: when defined, an invocation will be selected to start the migration
  benchmarking. Set environment variable  `AVA_MIGRATION_CALL_ID` to `r<limit>` (such as `r100`)
  to choose a random call, and to `<count>` to choose a specified call.
* `AVA_PRINT_TIMESTAMP`: when defined, APIs with `ava_time_me` will print the timestamps for the
  invocation's forwarding stages. The current sequence of an invocation is:
  
  ```
  Guestlib: before_marshal,
  Guestlib: before_send_command,
  Worker: after_receive_command,
  Worker: after_unmarshal,
  Worker: after_execution,
  Worker: after_marshal,
  Guestlib: after_receive_result.
  ```

The makefile generated by CAvA accepts a "release build" flag used as follows:
`make RELEASE=1` (or for the very lazy `make R=1`). When set this flag will enable optimization
and disable debugging. In release mode the build will:

* use `-O2`.
* disable `-g`.
* disable debug printing.
* disable assertions and many error check that should only happen if there is a bug in AvA.

This mode will almost certainly be faster (since it will allow the compiler to aggressively
optimize the primary data path). However, it will make debugging basically impossible, since
crashes will likely be due to data corruption instead of a nice assert.
The main reason for disabling `-g` is to remind us to rebuild in debug mode before trying to debug.

Publications
============

Yu, Hangchen, Arthur M. Peters, Amogh Akshintala, and Christopher J. Rossbach. "Automatic Virtualization of Accelerators." In Proceedings of the Workshop on Hot Topics in Operating Systems, pp. 58-65. ACM, 2019.

[Todo: add conference version]

Developers and contributors
===========================

| Name                    | Affiliation                 | Role           | Contact                |
|-------------------------|-----------------------------|----------------|------------------------|
| Hangchen Yu             | UT Austin                   | Main developer | hyu@cs.utexas.edu      |
| Arthur M. Peters        | UT Austin                   | Main developer | amp@cs.utexas.edu      |
| Amogh Akshintala        | UNC                         |                |                        |
| Tyler Hunt              | UT Austin                   |                |                        |
| Christopher J. Rossbach | UT Austin & VMware Research | Advisor        | rossbach@cs.utexas.edu |
