#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS3_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS3_UNIMPLEMENTED_H_
/* --------------- CUBLAS BLAS3 functions  ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                     cublasOperation_t transb, int m, int n, int k,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *B, int ldb,
                                                     const double *beta, /* host or device pointer */
                                                     double *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                     cublasOperation_t transb, int m, int n, int k,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                                                    cublasOperation_t transb, int m, int n, int k,
                                                    const cuComplex *alpha, /* host or device pointer */
                                                    const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                    const cuComplex *beta, /* host or device pointer */
                                                    cuComplex *C, int ldc) {
  ava_unsupported;
}
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa,
                                                      cublasOperation_t transb, int m, int n, int k,
                                                      const cuComplex *alpha, const void *A, cudaDataType Atype,
                                                      int lda, const void *B, cudaDataType Btype, int ldb,
                                                      const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                     cublasOperation_t transb, int m, int n, int k,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                     int ldb, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                                                    cublasOperation_t transb, int m, int n, int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */
                                                    const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                    int ldb, const cuDoubleComplex *beta, /* host or device pointer */
                                                    cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                                                  cublasOperation_t transb, int m, int n, int k,
                                                  const __half *alpha, /* host or device pointer */
                                                  const __half *A, int lda, const __half *B, int ldb,
                                                  const __half *beta, /* host or device pointer */
                                                  __half *C, int ldc) {
  ava_unsupported;
}

/* IO in FP16/FP32, computation in float */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                                    cublasOperation_t transb, int m, int n, int k,
                                                    const float *alpha, /* host or device pointer */
                                                    const void *A, cudaDataType Atype, int lda, const void *B,
                                                    cudaDataType Btype, int ldb,
                                                    const float *beta, /* host or device pointer */
                                                    void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                                   cublasOperation_t transb, int m, int n, int k,
                                                   const void *alpha, /* host or device pointer */
                                                   const void *A, cudaDataType Atype, int lda, const void *B,
                                                   cudaDataType Btype, int ldb,
                                                   const void *beta, /* host or device pointer */
                                                   void *C, cudaDataType Ctype, int ldc, cudaDataType computeType,
                                                   cublasGemmAlgo_t algo) {
  ava_unsupported;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                                    cublasOperation_t transb, int m, int n, int k,
                                                    const cuComplex *alpha, const void *A, cudaDataType Atype, int lda,
                                                    const void *B, cudaDataType Btype, int ldb, const cuComplex *beta,
                                                    void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa,
                                                          cublasOperation_t transb, cublasOperation_t transc, int m,
                                                          int n, int k, const unsigned char *A, int A_bias, int lda,
                                                          const unsigned char *B, int B_bias, int ldb, unsigned char *C,
                                                          int C_bias, int ldc, int C_mult, int C_shift) {
  ava_unsupported;
}

/* SYRK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, int n, int k,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda,
                                                     const float *beta, /* host or device pointer */
                                                     float *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, int n, int k,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda,
                                                     const double *beta, /* host or device pointer */
                                                     double *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, int n, int k,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, int n, int k,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda,
                                                     const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}
/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                    cublasOperation_t trans, int n, int k,
                                                    const cuComplex *alpha, /* host or device pointer */
                                                    const void *A, cudaDataType Atype, int lda,
                                                    const cuComplex *beta, /* host or device pointer */
                                                    void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k, const cuComplex *alpha,
                                                      const void *A, cudaDataType Atype, int lda, const cuComplex *beta,
                                                      void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

/* HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, int n, int k,
                                                     const float *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda,
                                                     const float *beta, /* host or device pointer */
                                                     cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, int n, int k,
                                                     const double *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda,
                                                     const double *beta, /* host or device pointer */
                                                     cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                    cublasOperation_t trans, int n, int k,
                                                    const float *alpha, /* host or device pointer */
                                                    const void *A, cudaDataType Atype, int lda,
                                                    const float *beta, /* host or device pointer */
                                                    void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k, const float *alpha,
                                                      const void *A, cudaDataType Atype, int lda, const float *beta,
                                                      void *C, cudaDataType Ctype, int ldc) {
  ava_unsupported;
}

/* SYR2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A, int lda, const float *B, int ldb,
                                                      const float *beta, /* host or device pointer */
                                                      float *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A, int lda, const double *B, int ldb,
                                                      const double *beta, /* host or device pointer */
                                                      double *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                      int ldb, const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}
/* HER2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                      const float *beta, /* host or device pointer */
                                                      cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                      cublasOperation_t trans, int n, int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                      int ldb, const double *beta, /* host or device pointer */
                                                      cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}
/* SYRKX : eXtended SYRK*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                   cublasOperation_t trans, int n, int k,
                                                   const float *alpha, /* host or device pointer */
                                                   const float *A, int lda, const float *B, int ldb,
                                                   const float *beta, /* host or device pointer */
                                                   float *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                   cublasOperation_t trans, int n, int k,
                                                   const double *alpha, /* host or device pointer */
                                                   const double *A, int lda, const double *B, int ldb,
                                                   const double *beta, /* host or device pointer */
                                                   double *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                   cublasOperation_t trans, int n, int k,
                                                   const cuComplex *alpha, /* host or device pointer */
                                                   const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                   const cuComplex *beta, /* host or device pointer */
                                                   cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                   cublasOperation_t trans, int n, int k,
                                                   const cuDoubleComplex *alpha, /* host or device pointer */
                                                   const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                                   const cuDoubleComplex *beta, /* host or device pointer */
                                                   cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}
/* HERKX : eXtended HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                   cublasOperation_t trans, int n, int k,
                                                   const cuComplex *alpha, /* host or device pointer */
                                                   const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                   const float *beta, /* host or device pointer */
                                                   cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                   cublasOperation_t trans, int n, int k,
                                                   const cuDoubleComplex *alpha, /* host or device pointer */
                                                   const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                                   const double *beta, /* host or device pointer */
                                                   cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}
/* SYMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, int m, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *B, int ldb,
                                                     const float *beta, /* host or device pointer */
                                                     float *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, int m, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *B, int ldb,
                                                     const double *beta, /* host or device pointer */
                                                     double *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                     int ldb, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

/* HEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                     int ldb, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

/* TRSM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, float *B, int ldb) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, double *B, int ldb) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, cuComplex *B, int ldb) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) {
  ava_unsupported;
}

/* TRMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *B, int ldb, float *C,
                                                     int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *B, int ldb, double *C,
                                                     int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                     cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo, cublasOperation_t trans,
                                                     cublasDiagType_t diag, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                                                     int ldb, cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}
/* BATCH GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const float *alpha, /* host or device pointer */
                                                         const float *const Aarray[], int lda,
                                                         const float *const Barray[], int ldb,
                                                         const float *beta, /* host or device pointer */
                                                         float *const Carray[], int ldc, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const double *alpha, /* host or device pointer */
                                                         const double *const Aarray[], int lda,
                                                         const double *const Barray[], int ldb,
                                                         const double *beta, /* host or device pointer */
                                                         double *const Carray[], int ldc, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const cuComplex *alpha, /* host or device pointer */
                                                         const cuComplex *const Aarray[], int lda,
                                                         const cuComplex *const Barray[], int ldb,
                                                         const cuComplex *beta, /* host or device pointer */
                                                         cuComplex *const Carray[], int ldc, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                           cublasOperation_t transb, int m, int n, int k,
                                                           const cuComplex *alpha, /* host or device pointer */
                                                           const cuComplex *const Aarray[], int lda,
                                                           const cuComplex *const Barray[], int ldb,
                                                           const cuComplex *beta, /* host or device pointer */
                                                           cuComplex *const Carray[], int ldc, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const cuDoubleComplex *alpha, /* host or device pointer */
                                                         const cuDoubleComplex *const Aarray[], int lda,
                                                         const cuDoubleComplex *const Barray[], int ldb,
                                                         const cuDoubleComplex *beta, /* host or device pointer */
                                                         cuDoubleComplex *const Carray[], int ldc, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void *alpha,                                                                   /* host or device pointer */
    const void *A, cudaDataType Atype, int lda, long long int strideA,                   /* purposely signed */
    const void *B, cudaDataType Btype, int ldb, long long int strideB, const void *beta, /* host or device pointer */
    void *C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const float *alpha,                                                /* host or device pointer */
    const float *A, int lda, long long int strideA,                    /* purposely signed */
    const float *B, int ldb, long long int strideB, const float *beta, /* host or device pointer */
    float *C, int ldc, long long int strideC, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const double *alpha,                                                 /* host or device pointer */
    const double *A, int lda, long long int strideA,                     /* purposely signed */
    const double *B, int ldb, long long int strideB, const double *beta, /* host or device pointer */
    double *C, int ldc, long long int strideC, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuComplex *alpha,                                                    /* host or device pointer */
    const cuComplex *A, int lda, long long int strideA,                        /* purposely signed */
    const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, /* host or device pointer */
    cuComplex *C, int ldc, long long int strideC, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuComplex *alpha,                                                    /* host or device pointer */
    const cuComplex *A, int lda, long long int strideA,                        /* purposely signed */
    const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, /* host or device pointer */
    cuComplex *C, int ldc, long long int strideC, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuDoubleComplex *alpha,                                                          /* host or device pointer */
    const cuDoubleComplex *A, int lda, long long int strideA,                              /* purposely signed */
    const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, /* host or device poi */
    cuDoubleComplex *C, int ldc, long long int strideC, int batchCount) {
  ava_unsupported;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS3_UNIMPLEMENTED_H_
