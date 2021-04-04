Demo AvA Manager
===================

This demo manager shows an example of a minimal AvA manager running on localhost:3333.
The demo manager runs a RPC service which receives `WorkerAssignRequest` from a guestlib,
spawns an API server every time and returns the API server's address back to the guestlib.

Start Manager
=============

Assume we are at `${AVA_BUILD_DIR}` and use CUDA API remoting as an example.

```Shell
cmake ../ava -DAVA_GEN_CUDA_SPEC=ON -DAVA_MANAGER_LEGACY=ON
make
./install/bin/legacy_manager --worker_path generated/cudadrv_nw/worker
```

The manager will start and listen at `0.0.0.0:3333`. It will spawn API servers
from the provided API server binary.

Then link the application to `libguestlib.so`.
The application will start with AvA's generated CUDA library loaded.

```Shell
LD_LIBRARY_PATH=${AVA_BUILD_DIR}/generated/cudadrv_nw ./cuda_program
```
