#ifndef __CUBLAS_CPP_H__
#define __CUBLAS_CPP_H__

typedef struct __half {
  unsigned short x;
} __half;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const __half *alpha, /* host or device pointer */
                                                         const __half *const Aarray[], int lda,
                                                         const __half *const Barray[], int ldb,
                                                         const __half *beta, /* host or device pointer */
                                                         __half *const Carray[], int ldc, int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const __half *alpha,                                                 /* host or device pointer */
    const __half *A, int lda, long long int strideA,                     /* purposely signed */
    const __half *B, int ldb, long long int strideB, const __half *beta, /* host or device pointer */
    __half *C, int ldc, long long int strideC, int batchCount);

#endif
