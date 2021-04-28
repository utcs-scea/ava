Install PyTorch
===============

Install PyTorch 1.8.1 from [pytorch/pytorch/tree/v1.8.1](https://github.com/pytorch/pytorch/tree/v1.8.1).

Build specification
===================

Configure AvA in `~/ava-build` with

```shell
cmake -DAVA_GEN_PYTORCH_SPEC=ON -DAVA_MANAGER_LEGACY=ON ../ava
make
```

Run benchmark
=============

Dump CUDA binaries for the benchmark (Generally speaking, in most cases, CUDA
binaries generated from a large PyTorch benchmark can be used for most other
benchmarks):

```shell
./install/bin/legacy_manager install/pt_dump/bin/worker
LD_LIBRARY_PATH=~/ava-build/install/pt_dump/lib/ python3 your_pytorch_benchmark.py
sudo mkdir /cuda_dumps
sudo cp /tmp/*.ava /cuda_dumps
```

> CUDA version: search `10.1` in `pt_dump.c` and `pt_opt.c`.
> Dump path: search `cuda_dumps` in `pt_opt.c`.

The CUDA dumps should be copied to both local (where benchmarks run) and remote
(where API servers run) servers.

Finally restart AvA manager and run the benchmark:

```shell
./install/bin/legacy_manager install/pt_opt/bin/worker
LD_LIBRARY_PATH=~/ava-build/install/pt_opt/lib/ python3 your_pytorch_benchmark.py
```
