Katana AvA Manager
==================

[Katana's AvA manager](https://github.com/KatanaGraph/katana/tree/master/libava)
shows a design of split manager and spawn daemon.

Start Components
================

Assume we are at `${AVA_BUILD_DIR}`.
Start the AvA manager by the following command (which runs the manager service
at port `3334`).
Note that the port `3333` is reserved by the manager in the current version, used
for guestlib's connection.

```Shell
./worker/katana/manager -m 0.0.0.0:3334
```

Open another terminal, start the AvA spawn daemon on the local machine (as
this example) or a remote GPU server:

```Shell
./worker/katana/daemon -f gpu.conf -w generated/cu_nw/worker -m 0.0.0.0:3334 -d 0.0.0.0:3335
```

The daemon's address can be set with `-d 0.0.0.0:3335`, and the API server's base port can be set
with `-b 4000`.
The daemon will display and register the GPU information in the manager.
The manager will also request it to spawn an API server pool on every GPU.

Then the application can be started by loading AvA's generated CUDA library.

```Shell
LD_LIBRARY_PATH=generated/cu_nw ./cuda_program
```
