Galvanic AvA Manager
===================

Galvanic AvA manager shows an example where the AvA manager is integrated into
the (Galvanic serverless) service, and only the spawn daemon is needed on the
compute nodes.

Start Components
================

Assume we are at `${AVA_BUILD_DIR}`.

Start Galvanic resource manager first before starting the AvA spawn daemon
on the local machine (as this example) or a remote GPU server:

```Shell
./worker/galvanic/daemon -f gpu.conf -w generated/cu_nw/worker -m 0.0.0.0:3334 -d 0.0.0.0:3335
```

The daemon's address can be set with `-d 0.0.0.0:3335`, and the API server's base port can be set
with `-b 4000`.
The daemon will display and register the GPU information in the resource manager
listening at `0.0.0.0:3334`.

To test the spawn daemon without Galvanic resource manager, start the AvA manager by the
following command (which runs the manager service at port `3334` and sets the API server
pool size to `0`).
Katana manager does not enable API server pool at this moment.
Note that the port `3333` is reserved by the manager in the current version, used
for guestlib's connection.

```Shell
./worker/galvanic/manager -m 0.0.0.0:3334 [-n 0]
```

Then the application can be started by loading AvA's generated CUDA library.

```Shell
LD_LIBRARY_PATH=generated/cu_nw ./cuda_program
```
