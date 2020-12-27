Demo AvA Manager
===================

This demo manager shows an example of a minimal AvA manager running on localhost:3333.
The demo manager runs a RPC service which receives `WorkerAssignRequest` from a guestlib,
spawns an API server every time and returns the API server's address back to the guestlib.

Start Manager
=============

Assume we are at `${AVA_BUILD_DIR}`.

```Shell
./worker/demo/manager generated/cudadrv_nw/worker
```

The manager will start and listen at `0.0.0.0:3333`. It will spawn API servers
from the provided API server binary.

Then the application can be started by loading AvA's generated CUDA library.

```Shell
LD_LIBRARY_PATH=generated/cudadrv_nw ./cuda_program
```
