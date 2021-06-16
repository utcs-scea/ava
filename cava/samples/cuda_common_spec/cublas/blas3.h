#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS3_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS3_H_
#include <cublas_api.h>
#include <cublas_v2.h>

/* --------------- CUBLAS BLAS3 functions  ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                          cublasOperation_t transb, int m, int n, int k,
                                                          const void *alpha, /* host or device pointer */
                                                          const void *const Aarray[], cudaDataType Atype, int lda,
                                                          const void *const Barray[], cudaDataType Btype, int ldb,
                                                          const void *beta, /* host or device pointer */
                                                          void *const Carray[], cudaDataType Ctype, int ldc,
                                                          int batchCount, cudaDataType computeType,
                                                          cublasGemmAlgo_t algo) {
  ava_argument(handle) ava_handle;
  // In tensorflow, Aarray, Barray and Carray are device memory
  ava_argument(Aarray) ava_opaque;
  ava_argument(Barray) ava_opaque;
  ava_argument(Carray) ava_opaque;
  // If they are host memory, use the following code:
  /*
  ava_argument(Aarray) {
      ava_in; ava_buffer(batchCount);
      ava_element {
          ava_in; ava_buffer(lda * __helper_a_last_dim_size(transa, k, m) * __helper_type_size(Atype));
      }
  }

  ava_argument(Barray) {
      ava_in; ava_buffer(batchCount);
      ava_element {
          ava_in; ava_buffer(ldb * __helper_b_last_dim_size(transb, k, n) * __helper_type_size(Btype));
      }
  }

  ava_argument(Carray) {
      ava_type_cast(void**);
      ava_out; ava_buffer(batchCount);
      ava_element {
          ava_out; ava_buffer(ldc * n * __helper_type_size(Ctype));
      }
  }
  */
  // TODO: figure out alpha and beta
  ava_argument(alpha) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_in;
    ava_buffer(1);
  }
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS3_H_
