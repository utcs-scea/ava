API Server
==========

GPU Provisioning
----------------

This shows an example of how to provision hardware resources in AvA.
When the API server is spawned by the spawn daemon (or AvA manager in the local
scheme), it is passed a few provisioning information via the environment variable
list (`execvpe(..., envp)`).

The following variable provisions three GPUs with different sizes of memory (in
bytes). The UUIDs may be the same as those GPUs may be provisioned from a same
physical one.

```shell
CUDA_VISIBLE_DEVICES=uuid1,uuid3   # Suppose uuid1 == uuid2
AVA_GPU_UUID=uuid1,uuid2,uuid3
AVA_GPU_MEMORY=mem1,mem2,mem3
```

Then the GPU "provisioner" provides a few APIs to map the provisioned GPUs to
the real hardware. Those APIs can be called from the AvA specification so forge
the returned GPU information.

```c
uint64_t ProvisionGpu::GetGpuMemory(unsigned gpu_id);
unsigned ProvisionGpu::GetGpuIndex(unsigned gpu_id);
```
