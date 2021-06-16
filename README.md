Automatic Virtualization of Accelerators
========================================

AvA is a research system for virtualizing general API-controlled
accelerators automatically, developed in SCEA lab at University of Texas
at Austin. AvA is prototyped on KVM and QEMU, compromising compatibility
with automation for classic API remoting systems, and introducing
hypervisor interposition for resource management and strong isolation.

This repository is the main codebase of AvA. We host customized Linux
kernel, QEMU, LLVM, and sets of benchmarks in separate repositories.

> We are refactoring AvA code for better extensibility and developer-friendliness.
> The refactoring breaks a few functionalities at this moment,
> but they are coming back.

### Setup code tree

``` bash
git clone git@github.com:utcs-scea/ava.git
cd ava
git submodule update --init --recursive
```

Documentations
--------------

* [Build and Setup](docs/build_and_setup.md)
* [Generate Code](docs/generate_code.md)
* [Guestlib Configurations](config/README.md)
* [Virtualize (Your Own) New APIs](docs/virtualize_new_api.md)
* [Introduction to LAPIS](cava/lapis.md)

Install
-------

### System requirements

AvA was fully tested on Ubuntu 18.04 (Linux 4.15) with GCC 7.5.0, Python 3.6.9,
Boost 1.71.x, cmake 3.19.1 and Protobuf 3.0-3.9.
The system also works in Ubuntu 16.04 with additional care of Python 3.6
and Clang-7 installation for CAvA scripts, but we do not maintain the support anymore.
We plan to migrate AvA to Ubuntu 20.04.

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

For the status of support, please check out [Build and Setup document](docs/build_and_setup.md#configuration).

Related repositories
--------------------

* Kernel: https://github.com/utcs-scea/ava-kvm
* QEMU: https://github.com/utcs-scea/ava-qemu
* LLVM: https://github.com/utcs-scea/ava-llvm
* Benchmarks: https://github.com/utcs-scea/ava-benchmarks

Publications
============

Yu, Hangchen, Arthur M. Peters, Amogh Akshintala, and Christopher J. Rossbach. "AvA: Accelerated Virtualization of Accelerators." In Proceedings of the 25th International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 807-825. ACM, 2020.

Yu, Hangchen, Arthur M. Peters, Amogh Akshintala, and Christopher J. Rossbach. "Automatic Virtualization of Accelerators." In Proceedings of the Workshop on Hot Topics in Operating Systems, pp. 58-65. ACM, 2019.

Developers and contributors
===========================

| Name                    | Affiliation                                | Role           | Contact                |
|-------------------------|--------------------------------------------|----------------|------------------------|
| Hangchen Yu             | Facebook & UT Austin                       | Main developer | hyu@cs.utexas.edu      |
| Arthur M. Peters        | Katana Graph & UT Austin                   | Main developer | amp@cs.utexas.edu      |
| Amogh Akshintala        | Facebook & UNC                             |                |                        |
| Zhiting Zhu             | UT Austin                                  |                |                        |
| Tyler Hunt              | Katana Graph & UT Austin                   |                |                        |
| Christopher J. Rossbach | UT Austin & Katana Graph & VMware Research | Advisor        | rossbach@cs.utexas.edu |
