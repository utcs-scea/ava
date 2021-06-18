#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL3_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL3_UNIMPLEMENTED_H_
#include <cusparse.h>

#include "cava/nightwatch/parser/c/nightwatch.h"
//##############################################################################
//# SPARSE LEVEL 3 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                                            int kb, int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
                                            const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA, const int blockSize, const float *B,
                                            const int ldb, const float *beta, float *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                                            int kb, int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
                                            const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA, const int blockSize, const double *B,
                                            const int ldb, const double *beta, double *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                                            int kb, int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                            const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA, const int blockSize, const cuComplex *B,
                                            const int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                                            int kb, int nnzb, const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                            const int blockSize, const cuDoubleComplex *B, const int ldb,
                                            const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz, const float *alpha,
                                            const float *A, int lda, const float *cscValB, const int *cscColPtrB,
                                            const int *cscRowIndB, const float *beta, float *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz, const double *alpha,
                                            const double *A, int lda, const double *cscValB, const int *cscColPtrB,
                                            const int *cscRowIndB, const double *beta, double *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                            const cuComplex *alpha, const cuComplex *A, int lda,
                                            const cuComplex *cscValB, const int *cscColPtrB, const int *cscRowIndB,
                                            const cuComplex *beta, cuComplex *C, int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                            const cuDoubleComplex *cscValB, const int *cscColPtrB,
                                            const int *cscRowIndB, const cuDoubleComplex *beta, cuDoubleComplex *C,
                                            int ldc) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsm2Info(csrsm2Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsm2Info(csrsm2Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle, csrsm2Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo,
                                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                                           int m, int nrhs, int nnz, const float *alpha,
                                                           const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                           const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                           const float *B, int ldb, csrsm2Info_t info,
                                                           cusparseSolvePolicy_t policy, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo,
                                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                                           int m, int nrhs, int nnz, const double *alpha,
                                                           const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                           const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                           const double *B, int ldb, csrsm2Info_t info,
                                                           cusparseSolvePolicy_t policy, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                      cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                      const float *alpha, const cusparseMatDescr_t descrA,
                                                      const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, const float *B, int ldb,
                                                      csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                      cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                      const double *alpha, const cusparseMatDescr_t descrA,
                                                      const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, const double *B, int ldb,
                                                      csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                      cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                      const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                                      const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, const cuComplex *B, int ldb,
                                                      csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                      cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                      const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb,
                                                      csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                   cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                   const float *alpha, const cusparseMatDescr_t descrA,
                                                   const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, float *B, int ldb, csrsm2Info_t info,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                   cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                   const double *alpha, const cusparseMatDescr_t descrA,
                                                   const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, double *B, int ldb, csrsm2Info_t info,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                   cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                   const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                                   const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, cuComplex *B, int ldb,
                                                   csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA,
                                                   cusparseOperation_t transB, int m, int nrhs, int nnz,
                                                   const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                                   const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, cuDoubleComplex *B, int ldb,
                                                   csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                        int n, int nnzb, const cusparseMatDescr_t descrA,
                                                        float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                        int n, int nnzb, const cusparseMatDescr_t descrA,
                                                        double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                        int n, int nnzb, const cusparseMatDescr_t descrA,
                                                        cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                        int n, int nnzb, const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                                           int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
                                                           float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                                           int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
                                                           double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                                           int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
                                                           cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                                           int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
                                                           cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                      int n, int nnzb, const cusparseMatDescr_t descrA,
                                                      const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                      int n, int nnzb, const cusparseMatDescr_t descrA,
                                                      const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                      int n, int nnzb, const cusparseMatDescr_t descrA,
                                                      const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                      int n, int nnzb, const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                   int n, int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
                                                   const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
                                                   const float *B, int ldb, float *X, int ldx,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                   int n, int nnzb, const double *alpha,
                                                   const cusparseMatDescr_t descrA, const double *bsrSortedVal,
                                                   const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                   int blockSize, bsrsm2Info_t info, const double *B, int ldb,
                                                   double *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
                                                   int n, int nnzb, const cuComplex *alpha,
                                                   const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
                                                   const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                   int blockSize, bsrsm2Info_t info, const cuComplex *B, int ldb,
                                                   cuComplex *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuDoubleComplex *B,
    int ldb, cuDoubleComplex *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL3_UNIMPLEMENTED_H_
