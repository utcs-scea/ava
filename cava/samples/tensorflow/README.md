Install TensorFlow
==================

Install TensorFlow 1.14 from [yuhc/tensorflow-cudart-dynam](https://github.com/yuhc/tensorflow-cudart-dynam).
Check out [BUILD.md](https://github.com/yuhc/tensorflow-cudart-dynam/blob/r1.14/BUILD.md) for build
instructions.

Build specification
===================

Configure AvA in `~/ava-build` with

```shell
cmake -DAVA_GEN_TF_SPEC=ON -DAVA_MANAGER_LEGACY=ON ../ava
make
```

Run benchmark
=============

Dump CUDA binaries for the benchmark (Generally speaking, in most cases, CUDA
binaries generated from a large TensorFlow benchmark can be used for most other
benchmarks):

```shell
./install/bin/legacy_manager install/tf_dump/bin/worker
LD_LIBRARY_PATH=~/ava-build/install/tf_dump/lib/ python3 your_tensorflow_benchmark.py
sudo mkdir /cuda_dumps
sudo cp /tmp/*.ava /cuda_dumps
```

> CUDA version: search `10.1` in `tf_dump.c` and `tf_opt.c`.
> Dump path: search `cuda_dumps` in `tf_opt.c`.

The CUDA dumps should be copied to both local (where benchmarks run) and remote
(where API servers run) servers.

Finally restart AvA manager and run the benchmark:

```shell
./install/bin/legacy_manager install/tf_opt/bin/worker
LD_LIBRARY_PATH=~/ava-build/install/tf_opt/lib/ python3 your_tensorflow_benchmark.py
```
