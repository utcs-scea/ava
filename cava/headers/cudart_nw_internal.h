#ifndef __CUDART_NW_INTERNAL_H__
#define __CUDART_NW_INTERNAL_H__

#ifdef __cplusplus
extern "C"
{
#endif

char CUDARTAPI
__cudaInitModule(void **fatCubinHandle);

void** CUDARTAPI
__cudaRegisterFatBinary(void *fatCubin);

void CUDARTAPI
__cudaUnregisterFatBinary(void **fatCubinHandle);

void CUDARTAPI
__cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize);

__host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3   gridDim,
                            dim3   blockDim,
                            size_t sharedMem, // CHECKME: default argument in header
                            void   *stream);

cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3   *gridDim,
                           dim3   *blockDim,
                           size_t *sharedMem,
                           void   *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
