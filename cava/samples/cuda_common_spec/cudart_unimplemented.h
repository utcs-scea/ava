#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUDART_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUDART_UNIMPLEMENTED_H_
#include <cuda_runtime_api.h>

__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value) { ava_unsupported; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                                                                   int *greatestPriority) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) { ava_unsupported; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) { ava_unsupported; }

// __host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event)
// {
//     ava_unsupported;
// }
//
// __host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle)
// {
//     ava_unsupported;
// }
//
// __host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
// {
//     ava_unsupported;
// }

// __host__ cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
// {
//     ava_unsupported;
// }
//
// __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr)
// {
//     ava_unsupported;
// }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr,
                                                                            int srcDevice, int dstDevice) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int *flags) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) { ava_unsupported; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                                                               unsigned int flags, int priority) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                                      unsigned int flags) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream,
                                                     enum cudaStreamCaptureStatus *pCaptureStatus) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream,
                                                        enum cudaStreamCaptureStatus *pCaptureStatus,
                                                        unsigned long long *pId) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaImportExternalMemory(cudaExternalMemory_t *extMem_out,
                                                        const struct cudaExternalMemoryHandleDesc *memHandleDesc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedBuffer(
    void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI
cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem,
                                          const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t extMem) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaImportExternalSemaphore(
    cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                                                           size_t sharedMem, cudaStream_t stream) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList,
                                                                      unsigned int numDevices,
                                                                      unsigned int flags __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr,
                                                                       int value) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size,
                                                                    unsigned int flags __dv(cudaMemAttachGlobal)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                                                 struct cudaExtent extent, unsigned int flags __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                                                        const struct cudaChannelFormatDesc *desc,
                                                        struct cudaExtent extent, unsigned int numLevels,
                                                        unsigned int flags __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray,
                                                          cudaMipmappedArray_const_t mipmappedArray,
                                                          unsigned int level) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) { ava_unsupported; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
                                                                    cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p,
                                                     cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent,
                                                unsigned int *flags, cudaArray_t array) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                                            size_t height, enum cudaMemcpyKind kind) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src,
                                                   size_t spitch, size_t width, size_t height,
                                                   enum cudaMemcpyKind kind) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset,
                                                     size_t hOffset, size_t width, size_t height,
                                                     enum cudaMemcpyKind kind) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                                        cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                                        size_t width, size_t height,
                                                        enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice,
                                                   size_t count, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                                                                    size_t spitch, size_t width, size_t height,
                                                                    enum cudaMemcpyKind kind,
                                                                    cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                                        const void *src, size_t spitch, size_t width, size_t height,
                                                        enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src,
                                                          size_t wOffset, size_t hOffset, size_t width, size_t height,
                                                          enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset,
                                                       enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset,
                                                         enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count,
                                                                  cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width,
                                                                    size_t height, cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value,
                                                                    struct cudaExtent extent,
                                                                    cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                                                    cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice,
                                             int device) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources,
                                                        cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources,
                                                          cudaStream_t stream __dv(0)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
                                                                    cudaGraphicsResource_t resource) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array,
                                                                     cudaGraphicsResource_t resource,
                                                                     unsigned int arrayIndex, unsigned int mipLevel) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                                                                           cudaGraphicsResource_t resource) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref,
                                               const void *devPtr, const struct cudaChannelFormatDesc *desc,
                                               size_t size __dv(UINT_MAX)) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemRangeGetAttribute(void *data, size_t dataSize,
                                                        enum cudaMemRangeAttribute attribute, const void *devPtr,
                                                        size_t count) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaMemRangeGetAttributes(void **data, size_t *dataSizes,
                                                         enum cudaMemRangeAttribute *attributes, size_t numAttributes,
                                                         const void *devPtr, size_t count) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaBindTexture2D(size_t *offset, const struct textureReference *texref,
                                                 const void *devPtr, const struct cudaChannelFormatDesc *desc,
                                                 size_t width, size_t height, size_t pitch) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array,
                                                      const struct cudaChannelFormatDesc *desc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToMipmappedArray(const struct textureReference *texref,
                                                               cudaMipmappedArray_const_t mipmappedArray,
                                                               const struct cudaChannelFormatDesc *desc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const void *symbol) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array,
                                                      const struct cudaChannelFormatDesc *desc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array) {
  ava_unsupported;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w,
                                                                      enum cudaChannelFormatKind f) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                                       const struct cudaResourceDesc *pResDesc,
                                                       const struct cudaTextureDesc *pTexDesc,
                                                       const struct cudaResourceViewDesc *pResViewDesc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                                                cudaTextureObject_t texObject) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc,
                                                               cudaTextureObject_t texObject) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc,
                                                                    cudaTextureObject_t texObject) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                                       const struct cudaResourceDesc *pResDesc) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                                                cudaSurfaceObject_t surfObject) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) { ava_unsupported; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                      const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                      const struct cudaKernelNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t node,
                                                            struct cudaKernelNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                                                            const struct cudaKernelNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                      const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                      const struct cudaMemcpy3DParms *pCopyParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,
                                                            struct cudaMemcpy3DParms *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node,
                                                            const struct cudaMemcpy3DParms *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                      const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                      const struct cudaMemsetParams *pMemsetParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,
                                                            struct cudaMemsetParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeSetParams(cudaGraphNode_t node,
                                                            const struct cudaMemsetParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                    const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                    const struct cudaHostNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphHostNodeGetParams(cudaGraphNode_t node,
                                                          struct cudaHostNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                                                          const struct cudaHostNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                          const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                          cudaGraph_t childGraph) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                     const cudaGraphNode_t *pDependencies, size_t numDependencies) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode,
                                                        cudaGraph_t clonedGraph) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes,
                                                     size_t *pNumRootNodes) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to,
                                                 size_t *numEdges) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies,
                                                            size_t *pNumDependencies) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes,
                                                              size_t *pNumDependentNodes) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
                                                        const cudaGraphNode_t *to, size_t numDependencies) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
                                                           const cudaGraphNode_t *to, size_t numDependencies) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphDestroyNode(cudaGraphNode_t node) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                                    cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                                                const struct cudaKernelNodeParams *pNodeParams) {
  ava_unsupported;
}

__host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec) { ava_unsupported; }

__host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) { ava_unsupported; }

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUDART_UNIMPLEMENTED_H_
