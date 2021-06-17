#ifndef __CUDART_NW_INTERNAL_H__
#define __CUDART_NW_INTERNAL_H__

#ifdef __cplusplus
extern "C" {
#endif

extern char CUDARTAPI __cudaInitModule(void **fatCubinHandle);

extern void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin);

extern void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle);

extern void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                                             const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                                             dim3 *bDim, dim3 *gDim, int *wSize);

extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                                        const char *deviceName, int ext, size_t size, int constant, int global);

__host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                            size_t sharedMem,  // CHECKME: default argument in header
                            void *stream);

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream);

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle);

extern void CUDARTAPI __cudaRegisterTexture(void **fatCubinHandle, const void *hostVar, const void **deviceAddress,
                                            const char *deviceName, int dim, int norm, int ext);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
