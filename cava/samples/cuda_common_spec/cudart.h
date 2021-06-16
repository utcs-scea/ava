#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDART_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDART_H_
#include <cuda_runtime_api.h>

__host__ __cudart_builtin__ const char *CUDARTAPI cudaGetErrorName(cudaError_t error) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
  ava_argument(pStream) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) { ava_argument(stream) ava_handle; }

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) { ava_argument(stream) ava_handle; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) { ava_argument(event) ava_handle; }

__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *_free, size_t *total) {
  ava_argument(_free) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(total) {
    ava_out;
    ava_buffer(1);
  }
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0),
                                                    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
  /* kind is always cudaMemcpyDeviceToHost */
  ava_argument(dst) {
    ava_out;
    ava_buffer(count);
  }
  ava_argument(symbol) ava_opaque;
}

#endif // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDART_H_
