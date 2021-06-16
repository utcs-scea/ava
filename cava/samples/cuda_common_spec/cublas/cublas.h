#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_H_

#include <cublas_api.h>
#include <cublas_v2.h>
/* CUDABLAS API */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate(cublasHandle_t *handle) {
  ava_disable_native_call;
  ava_argument(handle) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_handle; }
  }

  cublasStatus_t ret;
  if (ava_is_worker) {
    ret = __cublasCreate(handle);
    return ret;
  }
}

cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B,
                                            int ldb) {
  ava_argument(A) {
    ava_in;
    ava_buffer(rows * cols * elemSize);
  }

  ava_argument(B) { ava_opaque; }
}

cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B,
                                            int ldb) {
  ava_argument(A) { ava_opaque; }

  ava_argument(B) {
    ava_out;
    ava_buffer(rows * cols * elemSize);
  }
}

ava_begin_replacement;
EXPORTED CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle,
                                                                       cublasPointerMode_t *mode) {
  /* XXX seems ok for tensorflow but might be wrong !FIXME */
  *mode = CUBLAS_POINTER_MODE_HOST;
  return CUBLAS_STATUS_SUCCESS;
}

EXPORTED CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle,
                                                                       cublasPointerMode_t mode) {
  /* XXX seems ok for tensorflow but might be wrong ! FIXME */
  assert(mode == CUBLAS_POINTER_MODE_HOST && "mode == CUBLAS_POINTER_MODE_HOST");
  return CUBLAS_STATUS_SUCCESS;
}
ava_end_replacement;

#include "blas1_unimplemented.h"
#include "blas2_unimplemented.h"
#include "blas3.h"
#include "blas3_unimplemented.h"
#include "blas_like_ext.h"
#include "blas_like_ext_unimplemented.h"
#include "cublas_unimplemented.h"

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_H_