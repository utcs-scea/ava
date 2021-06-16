Generate Code
=============

## Generation Process

The code generation (aka building the specifications) is done via a separate script,
`generate.py`. The script is a wrapper of Cava, which configures all necessary Cava
arguments for building different supported specifications.

`generate.py` downloads the customized AvA-LLVM module into `$AVA_ROOT/llvm/build`
and unpacks the tarball. It then executes Cava to generate codes from the provided
specifications.

## Generator Script

To see the supported specifications, please run:

```shell
./generate.py -h
```

Here is a full list of the supported specifications, input after `-s`:

```shell
./generate.py -s cudadrv cudart demo gti ncsdk onnx_dump onnx_opt opencl pt_dump pt_opt qat test tf_c tf_dump tf_opt
```

### AvA Optimizations

We implement common optimization techniques in the AvA libraries and generate necessary
dependent codes using Cava, instead of injecting those optimization codes from specifications.
To enable those optimizations, run `generate.py` with `-O`:

```shell
./generate.py -s tf_opt -O batching
```

Cava will generate a macro `AVA_OPT_{CAPITALIZED_OPTIMIZATION_NAME}` and pass it to both
guestlib and API server at the compile time. In the above example, the guestlib's and API
server's compile option will contain `-DAVA_OPT_BATCHING` in the generated CMakeLists.txt.

#### Batching

TO BE COMPLETED