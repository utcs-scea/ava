#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS_LIKE_EXT_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS_LIKE_EXT_H_
#include <cublas_v2.h>
#include <cublas_api.h>

/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                  cublasOperation_t transb, int m, int n,
                                                  const float *alpha, /* host or device pointer */
                                                  const float *A, int lda,
                                                  const float *beta, /* host or device pointer */
                                                  const float *B, int ldb, float *C, int ldc) {
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(A) ava_opaque;
  ava_argument(beta) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(B) ava_opaque;
  ava_argument(C) ava_opaque;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(streamId) ava_handle;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal(cublasHandle_t handle, int n,
                                                  const float *alpha, /* host or device pointer */
                                                  float *x, int incx) {
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(x) ava_opaque;
}


#endif // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS_LIKE_EXT_H_
