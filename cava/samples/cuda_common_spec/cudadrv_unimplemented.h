#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDADRV_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDADRV_UNIMPLEMENTED_H_
#include <cuda.h>

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) { ava_unsupported; }

CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) { ava_unsupported; }

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) { ava_unsupported; }

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) { ava_unsupported; }

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
  ava_unsupported;
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) { ava_unsupported; }

CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) { ava_unsupported; }

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize,
                                                             size_t dynamicSMemSize) {
  ava_argument(numBlocks) {
    ava_out;
    ava_buffer(1);
  }

  ava_unsupported;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize,
                                                                      size_t dynamicSMemSize, unsigned int flags) {
  ava_argument(numBlocks) {
    ava_out;
    ava_buffer(1);
  }

  ava_unsupported;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDADRV_UNIMPLEMENTED_H_
