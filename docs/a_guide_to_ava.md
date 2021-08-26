# An RA's guide to AvA
By [Esther Yoon](mailto:estheryoon@utexas.edu), @[estherlyoon](https://github.com/estherlyoon), original [Google docs link](https://docs.google.com/document/d/1KZUMMJW46DLvdOd_6pL865wBWFmuFy-j3dR2DB3wp5Y/edit?usp=sharing)

> Esther is an undergraduate student in the Department of Computer Sciences and [Turing Scholars Program](https://www.cs.utexas.edu/turing-scholars) at UT Austin.
> She works as a research assistant in the SCEA lab on an AvA-related research project.
 
Before you do anything, you should read the [paper](https://oscarlab.github.io/papers/ava-asplos20.pdf) and check out the [repo](https://github.com/utcs-scea/ava). What’s written here is not a comprehensive summary of how AvA works, rather notes that I think are helpful when interacting with AvA for the first time. Parts of this document also assume that you’re running on one of the csres machines (zemaitis, languedoc, santacruz).


## Explanation of compiling and running

There are build instructions in the github repository, however, I thought it might be helpful to provide some pointers on parts of the process.

_Terminology_

Below are brief explanations of important terms that I’ll use periodically. Some, again, are in the paper and repo already, but I think it helps to clarify a few.

_Manager_: the manager facilitates connections between the application and the virtual environment, and spawns workers (by forking and execing) to process application-side calls. It can also enforce policies such as scheduling multiple workers to multiple GPUs. To my knowledge, one worker is paired with one application. It might get more complicated in a multi-threaded application, which I don’t have experience with.

_Worker_: a worker is on the remote side of AvA, and is responsible for executing API calls on behalf of the application. At start time, a worker will wait for a guestlib connection, perform initialization like loading in necessary binaries, then launch it’s command handlers and run until the internal handler exits.

_Guestlib_: the guestlib is on the application side of AvA, and is what an application links to through the `LD_LIBRARY_PATH` environment variable (I often think of “application” and “guestlib” as interchangeable terms). This layer interposes API calls, which are then forwarded to a remote worker.

_Spec / Specification_: The hand-written code used by CAvA to create the generated code for a specific API. Examples of pre-written specifications can be found in `cava/samples`. You can add additional API implementations and write your own specifications.

_Generated Code_: The code produced by running `generate.py` that processes API calls for a particular specification, which will be found in `cava/<spec-name>_nw`. I heard this referred to as a “dump” a few times, which is not to be confused with the fat binary “dump files” that are discussed further down this document.

_Build tips_

You might think about using the `cudart` spec for all your CUDA-related needs, but _don’t use this_, as it’s not well tested or developed. You should be using the `onnx_dump` and `onnx_opt` specs, as these have been used for most development and have many more API calls implemented.

If you’re frequently modifying parts of AvA that don’t affect the generated code, don’t waste time repeatedly running the generate.py script part of the build. We had one task that ran the `generate.py` and `check_cpp_format.py` scripts, and another that contained the steps for compiling everything in the build directory.

You have the option of building with “debug” or “release” mode with the cmake option `-DCMAKE_BUILD_TYPE=Debug` (or Release). Debug mode is compiled on O0 and produces more output, such as information on API calls being made, while release mode is compiled on O3 and has less output.

When using the lab machines, it can make your life easier to run in a container. Instructions to deploy it [here](https://github.com/utcs-scea/ava/tree/master/tools/docker).


## Misc. good-to-know things 

_Fatbins and preloading_

CUDA code is usually compiled in two steps, from device code → PTX virtual assembly → binary code. The second step can be JIT compiled by the CUDA driver, but this causes runtime overheads. One way this cost is mitigated is through the use of fat binaries (AKA fatbins, cubins), files which store the binary code for a specific architecture(s). See this [NVIDIA article](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/) for details.

`cudaRegisterFatBinary` and `cudaRegisterFunction` are used to do preloading. These are internal functions not meant to be called by the programmer, and instead will be injected by nvcc into the source code in order to register the compiled cubin with the runtime. AvA performs the function of preloading for CUDA runtimes differently depending on which specification you use.

_dump v. opt_

There are two available ONNX Runtime specifications, denoted by the suffixes “dump” and “opt”. The difference between these has to do with the aforementioned fatbin preloading.

In the dump spec, fat binaries are _generated _every time an application is run, with all the information needed to run the program stored in `.ava` files (which we also refer to as _dump files_). `__cudaRegisterFunction` and `__cudaRegisterFatbinary` are called, where normal loading is done on the guestlib side and dump files are written to `/tmp` on the worker side.

The opt in `onnx_opt` indicates an optimized version of the spec. Instead of generating dump files, this spec expects them to be pre-generated, thus reducing the time spent on initialization tasks. The loading happens when the guestlib and worker call `ava_preload_cubin_guestlib` and `ava_load_cubin_worker`, respectively. The default directory searched for dump files is `/cuda_dumps`. Our [fork](https://github.com/hfingler/ava) of AvA was modified to accept an environment variable on the guestlib side to specify the dump file directory. If you’d like to incorporate a similar feature, feel free to contact one of the contributors. 

_Handlers_

There are two running handlers that serve different purposes.

The command handler lives inside the guestlib’s and worker’s generated code, denoted by `__handle_command_&lt;spec-name>`, that is responsible for communicating and executing API calls. The `ava_endpoint` is often operated on within the handler and generated code in general, doing things like adding and removing calls. See `common/endpoint_lib.hpp` for more applications of `ava_endpoint`.

Additionally, there is the `internal_api_handler`, located in `common/cmd_handler.cpp`. This handler is responsible for processing events unrelated to API calls, such as triggering migration across GPUs and system shutdown.

Both handlers are instantiated within the guestlib: see `guestlib/init.cpp`. Looking through how each is initialized and processes events can be a helpful exercise.


## Debugging and output

Output redirected to `AVA_DEBUG` is written to file(s) in /tmp, the names of which are output on both the worker and guestlib side at initialization time (worker and guestlib logs are distinct). The log files names can be seen in this line:


```
To check the state of AvA remoting progress, use `tail -f /tmp/file2SKe5P`
```


You can collect timing information by enabling the `AVA_ENABLE_STAT` macro when generating onnx_opt. See instructions near the top of the file `cava/samples/onnxruntime/onnx_opt.cpp`
