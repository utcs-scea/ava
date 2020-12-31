Virtualize New API
==================

This tutorial instructs how to virtualize a new API with AvA using the demo API
as the example (API name: `demo`).

Basic cmake files
-----------

### Edit `ava/CMakeLists.txt`

We use cmake to configure our build. Add the following lines to `ava/CMakeLists.txt`
so that we can configure to build the new API virtualization:

```
set(AVA_GEN_DEMO_SPEC OFF CACHE BOOL "Enable demo specification")
message(STATUS "Build demo specification: ${AVA_GEN_DEMO_SPEC}")
```

Add the following line to `ava-spec` external project's `CMAKE_CACHE_ARGS`:

```
-DAVA_GEN_DEMO_SPEC:BOOL=${AVA_GEN_DEMO_SPEC}
```

### Edit `ava/cava/CMakeList.txt`

At the end of the file, add:

```
message(STATUS "Compile demo specification: ${AVA_GEN_DEMO_SPEC}")
if(AVA_GEN_DEMO_SPEC)
  include(samples/demo/CMakeLists.txt)
endif()
```

As it shows, we will put our new specification and corresponding CMakeLists file
in `ava/cava/samples/demo`.

Writing specification
---------------------

> We will write a more detailed tutorial on how to write API specifications
> ([Writing a Specification](writing_a_specification.md)).
> For now, [ava/cava/lapis.md](../cava/lapis.md) is the best documentation that we have for writing specifications.

### Specification

Write your first specification!

Create a `demo.c` under `ava/cava/samples/demo`.
Define the API name, version, identifier and number at the top of the specification:

```cpp
ava_name("DEMO");
ava_version("0.0.1");
ava_identifier(DEMO);
ava_number(1);
```

The identifier is the prefix of the generated files and directory (`demo_nw`).
The API number is the ID used to differentiate the virtualized APIs in the system, which is optional.

Then we need to provide C (`ava_cflags`) or C++ flags (`ava_cxxflags`) for compiling the generated code.
It usually tells the compiler where to find the included headers (e.g., `-I/usr/local/cuda-10.1/include`)
as well as other compilation options.
We can specify which libraries should be linked via `ava_libs` (e.g., `-lcuda`).

The workspace is in `ava/cava/samples/demo_nw`.
In this example, we defined the libdemo header in `ava/cava/headers/demo.h`, so we use that header
search path in the cflags and include the header file:

```cpp
ava_cflags(-I../headers);
#include <demo.h>
```

> Our specification compiler, cava, can be used to generate a preliminary specification from the
> specification we just edited.
> In `ava/cava`, run
>
> ```shell
> ./nwcc samples/demo/demo.c -Iheaders --dump
> ```
>
> which will print the specification to stdout.
> This feature may help when you are virtualizing a lot of APIs.
> But the annotations are generated from templates and require human refinition.

We will use `fprintf` in the specification to print logs, which are considered as an utility:

```
ava_begin_utility;
#include <stdio.h>
ava_end_utility;
```

At last, we declare the APIs from `demo.h` to be virtualized:

```
int ava_test_api(int x);
```

`ava_test_api` only takes a simple scalar parameter which needs no special annotation.
So it is good to go.

> In this example, we do not really write a demo library (i.e., libdemo).
> Instead, we define `ava_test_api` inside the specification.
> Check out `ava/cava/samples/demo/demo.c` for more details.

### CMakeLists


Manager cmake files
-------------------

The CMakeLists.txt for the specification can be more complicated than other cmake files.
Fortunately, one can copy the existing specifications' CMakeLists.txt and make simple modifications.
For example, starting from `ava/cava/samples/demo/CMakeLists.txt`:

1. Replace `demo` with the new API's identifier.
2. Change the parameters given to cava in the `nwcc` step.
3. Create symbolic links to `libguestlib.so` in the `link` step.

### Edit `ava/CMakeLists.txt`

Add the following lines to `ava/CMakeLists.txt` so that we can configure to build
the new API server manager:

```
set(AVA_MANAGER_DEMO OFF CACHE BOOL "Build demo manager")
message(STATUS "Build demo manager: ${AVA_MANAGER_DEMO}")
```

Add the following line to `ava-manager` external project's `CMAKE_CACHE_ARGS`:

```
-DAVA_MANAGER_DEMO:BOOL=${AVA_MANAGER_DEMO}
```

### Edit `ava/worker/CMakeLists.txt`

Add the following lines to the end of the file:

```
if(AVA_MANAGER_DEMO)
  add_subdirectory(demo)
endif()
```

Writing AvA manager
-------------------

A developer can write any AvA manager according their needs, as long as the manager
satisfies the following rules:

1. The manager receives a `WorkerAssignRequest` request (defined in `ava/proto/manager_service.proto`)
   from guestlib. The request is serialized and delimited:
   the first 4 bytes is the length (L) of the serialized request, and the next L bytes
   is the serialized request.
2. The manager spawns API servers with any kind of policies or behaviors. After that,
   the manager packs the spawned API server's address (IP:port) into a `WorkerAssignReply`
   response, serializes and delimits the response in the same way as the request.

To simplify this process, we defined a `ava_manager::ManagerServiceServerBase` class
in `ava/worker/include/manager_service.h`.
In this example, The developer creates `DemoManager` class inheriting this base class
and starts a basic request handling service by:

```cpp
DemoManager manager(kDefaultManagerPort, kDefaultWorkerPortBase, argv[1]);
manager.RunServer();
```

The developer can overload `HandleRequest` and `SpawnWorker` methods in the derived
class to enforce more complex scheduling policies.

The AvA manager can have multiple components built into separate binaries. The master
code must enforce the above rules, while there is no restriction of how the components
communicate internally (e.g., they can talk with gRPC instead of Boost::asio).

`ava/worker/demo/CMakeLists.txt` shows a minimal CMakeLists file for building an
AvA manager.

Build virtualization components
-------------------------------

At the root of the build directory, run:

```shell
cmake ../ava -DAVA_GEN_DEMO_SPEC=ON -DAVA_MANAGER_DEMO=ON
make
```

The command will generate virtualization codes from the demo API specification
into `ava/cava/demo_nw`.
The codes are compiled in `build/generated/demo_nw` and copied into `build/install/demo_nw`.
The demo manager is compiled in `build/worker/demo` and copied into `build/install/bin`.

