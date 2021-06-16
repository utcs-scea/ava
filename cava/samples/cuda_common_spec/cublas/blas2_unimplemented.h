#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS2_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS2_UNIMPLEMENTED_H_
#include <cublas_v2.h>
#include <cublas_api.h>

/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *x, int incx,
                                                     const float *beta, /* host or device pointer */
                                                     float *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *x, int incx,
                                                     const double *beta, /* host or device pointer */
                                                     double *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                     int incx, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}
/* GBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     int kl, int ku, const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *x, int incx,
                                                     const float *beta, /* host or device pointer */
                                                     float *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     int kl, int ku, const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *x, int incx,
                                                     const double *beta, /* host or device pointer */
                                                     double *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     int kl, int ku,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                     int kl, int ku,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                     int incx, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

/* TRMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const float *A, int lda, float *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const double *A, int lda, double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuComplex *A, int lda, cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

/* TBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const float *A, int lda, float *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const double *A, int lda, double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const cuComplex *A, int lda, cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

/* TPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const float *AP, float *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const double *AP, double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuComplex *AP, cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

/* TRSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const float *A, int lda, float *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const double *A, int lda, double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuComplex *A, int lda, cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

/* TPSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const float *AP, float *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const double *AP, double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuComplex *AP, cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                     const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

/* TBSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const float *A, int lda, float *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const double *A, int lda, double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const cuComplex *A, int lda, cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                     cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                     const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

/* SYMV/HEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *x, int incx,
                                                     const float *beta, /* host or device pointer */
                                                     float *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *x, int incx,
                                                     const double *beta, /* host or device pointer */
                                                     double *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                     int incx, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                     int incx, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

/* SBMV/HBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *x, int incx,
                                                     const float *beta, /* host or device pointer */
                                                     float *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *A, int lda, const double *x, int incx,
                                                     const double *beta, /* host or device pointer */
                                                     double *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                     int incx, const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

/* SPMV/HPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *AP, const float *x, int incx,
                                                     const float *beta, /* host or device pointer */
                                                     float *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *AP, const double *x, int incx,
                                                     const double *beta, /* host or device pointer */
                                                     double *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *AP, const cuComplex *x, int incx,
                                                     const cuComplex *beta, /* host or device pointer */
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx,
                                                     const cuDoubleComplex *beta, /* host or device pointer */
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

/* GER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle, int m, int n,
                                                    const float *alpha, /* host or device pointer */
                                                    const float *x, int incx, const float *y, int incy, float *A,
                                                    int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle, int m, int n,
                                                    const double *alpha, /* host or device pointer */
                                                    const double *x, int incx, const double *y, int incy, double *A,
                                                    int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                     cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle, int m, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                     cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                                                     int incy, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle, int m, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                                                     int incy, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

/* SYR/HER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const float *alpha, /* host or device pointer */
                                                    const float *x, int incx, float *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const double *alpha, /* host or device pointer */
                                                    const double *x, int incx, double *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const cuComplex *alpha, /* host or device pointer */
                                                    const cuComplex *x, int incx, cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */
                                                    const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const float *alpha, /* host or device pointer */
                                                    const cuComplex *x, int incx, cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const double *alpha, /* host or device pointer */
                                                    const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

/* SPR/HPR */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const float *alpha, /* host or device pointer */
                                                    const float *x, int incx, float *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const double *alpha, /* host or device pointer */
                                                    const double *x, int incx, double *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const float *alpha, /* host or device pointer */
                                                    const cuComplex *x, int incx, cuComplex *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                    const double *alpha, /* host or device pointer */
                                                    const cuDoubleComplex *x, int incx, cuDoubleComplex *AP) {
  ava_unsupported;
}

/* SYR2/HER2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *x, int incx, const float *y, int incy, float *A,
                                                     int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *x, int incx, const double *y, int incy, double *A,
                                                     int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                     cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                                                     int incy, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                     cuComplex *A, int lda) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                                                     int incy, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

/* SPR2/HPR2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *x, int incx, const float *y, int incy, float *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *x, int incx, const double *y, int incy, double *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                     cuComplex *AP) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                                                     int incy, cuDoubleComplex *AP) {
  ava_unsupported;
}

#endif // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS2_UNIMPLEMENTED_H_
