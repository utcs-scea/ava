#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_UNIMPLEMENTED_H_
/* CUDABLAS API */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr,
                                                            const char *logFileName) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetLoggerCallback(cublasLogCallback userCallback) { ava_unsupported; }

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetLoggerCallback(cublasLogCallback *userCallback) { ava_unsupported; }

cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy) {
  ava_unsupported;
}

cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) { ava_unsupported; }

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) { ava_unsupported; }

cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B,
                                                 int ldb, cudaStream_t stream) {
  ava_unsupported;
}

cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B,
                                                 int ldb, cudaStream_t stream) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const __half *alpha, /* host or device pointer */
                                                         const __half *const Aarray[], int lda,
                                                         const __half *const Barray[], int ldb,
                                                         const __half *beta, /* host or device pointer */
                                                         __half *const Carray[], int ldc, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const __half *alpha,                                                 /* host or device pointer */
    const __half *A, int lda, long long int strideA,                     /* purposely signed */
    const __half *B, int ldb, long long int strideB, const __half *beta, /* host or device pointer */
    __half *C, int ldc, long long int strideC, int batchCount) {
  ava_unsupported;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_UNIMPLEMENTED_H_
