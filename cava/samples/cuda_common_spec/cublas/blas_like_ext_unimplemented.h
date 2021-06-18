#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS_LIKE_EXT_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS_LIKE_EXT_UNIMPLEMENTED_H_
#include "cava/nightwatch/parser/c/nightwatch.h"
/* ---------------- CUBLAS BLAS-like extension ---------------- */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                  cublasOperation_t transb, int m, int n,
                                                  const double *alpha, /* host or device pointer */
                                                  const double *A, int lda,
                                                  const double *beta, /* host or device pointer */
                                                  const double *B, int ldb, double *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                  cublasOperation_t transb, int m, int n,
                                                  const cuComplex *alpha, /* host or device pointer */
                                                  const cuComplex *A, int lda,
                                                  const cuComplex *beta, /* host or device pointer */
                                                  const cuComplex *B, int ldb, cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                  cublasOperation_t transb, int m, int n,
                                                  const cuDoubleComplex *alpha, /* host or device pointer */
                                                  const cuDoubleComplex *A, int lda,
                                                  const cuDoubleComplex *beta, /* host or device pointer */
                                                  const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

/* Batched LU - GETRF*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle, int n,
                                                          float *const A[], /*Device pointer*/
                                                          int lda, int *P,  /*Device Pointer*/
                                                          int *info,        /*Device Pointer*/
                                                          int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle, int n,
                                                          double *const A[], /*Device pointer*/
                                                          int lda, int *P,   /*Device Pointer*/
                                                          int *info,         /*Device Pointer*/
                                                          int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle, int n,
                                                          cuComplex *const A[], /*Device pointer*/
                                                          int lda, int *P,      /*Device Pointer*/
                                                          int *info,            /*Device Pointer*/
                                                          int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle, int n,
                                                          cuDoubleComplex *const A[], /*Device pointer*/
                                                          int lda, int *P,            /*Device Pointer*/
                                                          int *info,                  /*Device Pointer*/
                                                          int batchSize) {
  ava_unsupported;
}

/* Batched inversion based on LU factorization from getrf */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle, int n,
                                                          const float *const A[], /*Device pointer*/
                                                          int lda, const int *P,  /*Device pointer*/
                                                          float *const C[],       /*Device pointer*/
                                                          int ldc, int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle, int n,
                                                          const double *const A[], /*Device pointer*/
                                                          int lda, const int *P,   /*Device pointer*/
                                                          double *const C[],       /*Device pointer*/
                                                          int ldc, int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle, int n,
                                                          const cuComplex *const A[], /*Device pointer*/
                                                          int lda, const int *P,      /*Device pointer*/
                                                          cuComplex *const C[],       /*Device pointer*/
                                                          int ldc, int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle, int n,
                                                          const cuDoubleComplex *const A[], /*Device pointer*/
                                                          int lda, const int *P,            /*Device pointer*/
                                                          cuDoubleComplex *const C[],       /*Device pointer*/
                                                          int ldc, int *info, int batchSize) {
  ava_unsupported;
}

/* Batched solver based on LU factorization from getrf */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                          int nrhs, const float *const Aarray[], int lda,
                                                          const int *devIpiv, float *const Barray[], int ldb, int *info,
                                                          int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                          int nrhs, const double *const Aarray[], int lda,
                                                          const int *devIpiv, double *const Barray[], int ldb,
                                                          int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                          int nrhs, const cuComplex *const Aarray[], int lda,
                                                          const int *devIpiv, cuComplex *const Barray[], int ldb,
                                                          int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                          int nrhs, const cuDoubleComplex *const Aarray[], int lda,
                                                          const int *devIpiv, cuDoubleComplex *const Barray[], int ldb,
                                                          int *info, int batchSize) {
  ava_unsupported;
}

/* TRSM - Batched Triangular Solver */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, int m, int n,
                                                         const float *alpha, /*Host or Device Pointer*/
                                                         const float *const A[], int lda, float *const B[], int ldb,
                                                         int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, int m, int n,
                                                         const double *alpha, /*Host or Device Pointer*/
                                                         const double *const A[], int lda, double *const B[], int ldb,
                                                         int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, int m, int n,
                                                         const cuComplex *alpha, /*Host or Device Pointer*/
                                                         const cuComplex *const A[], int lda, cuComplex *const B[],
                                                         int ldb, int batchCount) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, int m, int n,
                                                         const cuDoubleComplex *alpha, /*Host or Device Pointer*/
                                                         const cuDoubleComplex *const A[], int lda,
                                                         cuDoubleComplex *const B[], int ldb, int batchCount) {
  ava_unsupported;
}

/* Batched - MATINV*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle, int n,
                                                           const float *const A[],       /*Device pointer*/
                                                           int lda, float *const Ainv[], /*Device pointer*/
                                                           int lda_inv, int *info,       /*Device Pointer*/
                                                           int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle, int n,
                                                           const double *const A[],       /*Device pointer*/
                                                           int lda, double *const Ainv[], /*Device pointer*/
                                                           int lda_inv, int *info,        /*Device Pointer*/
                                                           int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle, int n,
                                                           const cuComplex *const A[],       /*Device pointer*/
                                                           int lda, cuComplex *const Ainv[], /*Device pointer*/
                                                           int lda_inv, int *info,           /*Device Pointer*/
                                                           int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle, int n,
                                                           const cuDoubleComplex *const A[],       /*Device pointer*/
                                                           int lda, cuDoubleComplex *const Ainv[], /*Device pointer*/
                                                           int lda_inv, int *info,                 /*Device Pointer*/
                                                           int batchSize) {
  ava_unsupported;
}

/* Batch QR Factorization */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                          float *const Aarray[],            /*Device pointer*/
                                                          int lda, float *const TauArray[], /*Device pointer*/
                                                          int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                          double *const Aarray[],            /*Device pointer*/
                                                          int lda, double *const TauArray[], /*Device pointer*/
                                                          int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                          cuComplex *const Aarray[],            /*Device pointer*/
                                                          int lda, cuComplex *const TauArray[], /*Device pointer*/
                                                          int *info, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                          cuDoubleComplex *const Aarray[],            /*Device pointer*/
                                                          int lda, cuDoubleComplex *const TauArray[], /*Device pointer*/
                                                          int *info, int batchSize) {
  ava_unsupported;
}
/* Least Square Min only m >= n and Non-transpose supported */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                         int nrhs, float *const Aarray[],       /*Device pointer*/
                                                         int lda, float *const Carray[],        /*Device pointer*/
                                                         int ldc, int *info, int *devInfoArray, /*Device pointer*/
                                                         int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                         int nrhs, double *const Aarray[],      /*Device pointer*/
                                                         int lda, double *const Carray[],       /*Device pointer*/
                                                         int ldc, int *info, int *devInfoArray, /*Device pointer*/
                                                         int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                         int nrhs, cuComplex *const Aarray[], /*Device pointer*/
                                                         int lda, cuComplex *const Carray[],  /*Device pointer*/
                                                         int ldc, int *info, int *devInfoArray, int batchSize) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                         int nrhs, cuDoubleComplex *const Aarray[], /*Device pointer*/
                                                         int lda, cuDoubleComplex *const Carray[],  /*Device pointer*/
                                                         int ldc, int *info, int *devInfoArray, int batchSize) {
  ava_unsupported;
}
/* DGMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                  const float *A, int lda, const float *x, int incx, float *C,
                                                  int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                  const double *A, int lda, const double *x, int incx, double *C,
                                                  int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                  const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                  cuComplex *C, int ldc) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                  const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                                                  cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

/* TPTTR : Triangular Pack format to Triangular format */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *AP,
                                                   float *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                   const double *AP, double *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                   const cuComplex *AP, cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                   const cuDoubleComplex *AP, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}
/* TRTTP : Triangular format to Triangular Pack format */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *A,
                                                   int lda, float *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *A,
                                                   int lda, double *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                   const cuComplex *A, int lda, cuComplex *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                   const cuDoubleComplex *A, int lda, cuDoubleComplex *AP) {
  ava_unsupported;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS_LIKE_EXT_UNIMPLEMENTED_H_
