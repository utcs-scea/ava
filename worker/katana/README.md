Katana AvA Manager
==================

[Katana's AvA manager](https://github.com/KatanaGraph/katana/tree/master/libava)
shows a design of split manager and spawn daemon.

Start Components
================

Assume we are at `${AVA_BUILD_DIR}`.
Start the AvA manager by the following command (which runs the manager service
at port `3334` and sets the API server pool size to `2`).
Note that the port `3333` is reserved by the manager in the current version, used
for guestlib's connection.

```Shell
./worker/katana/manager -m 0.0.0.0:3334 -n 2
```

Open another terminal, start the AvA spawn daemon on the local machine (as
this example) or a remote GPU server:

```Shell
./libava/manager/spawn_daemon -f gpu.conf -w libava/generated/worker -m 0.0.0.0:3334 -d 0.0.0.0:3335
```

The daemon's address can be set with `-d 0.0.0.0:3335`, and the API server's base port can be set
with `-b 4000`.
The daemon will display and register the GPU information in the manager.
The manager will also request it to spawn an API server pool on every GPU.

Then the application can be started by loading AvA's generated CUDA library.
In the build directory (`${KATANA_BUILD_DIR}`):

```Shell
LD_LIBRARY_PATH=libava/generated:libava/ava-bin/third_party/grpc/lib \
./lonestar/analytics/pagerank/pagerank-gpu inputs/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles
```
