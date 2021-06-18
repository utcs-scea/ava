#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_MATRIX_SORTING_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_MATRIX_SORTING_UNIMPLEMENTED_H_
#include <cusparse.h>

#include "cava/nightwatch/parser/c/nightwatch.h"
//##############################################################################
//# SPARSE MATRIX SORTING
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int *p) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                            const int *cooRowsA, const int *cooColsA,
                                                            size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int *cooRowsA,
                                                   int *cooColsA, int *P, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz, int *cooRowsA,
                                                      int *cooColsA, int *P, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                            const int *csrRowPtrA, const int *csrColIndA,
                                                            size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz,
                                              const cusparseMatDescr_t descrA, const int *csrRowPtrA, int *csrColIndA,
                                              int *P, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                            const int *cscColPtrA, const int *cscRowIndA,
                                                            size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz,
                                              const cusparseMatDescr_t descrA, const int *cscColPtrA, int *cscRowIndA,
                                              int *P, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                             float *csrVal, const int *csrRowPtr, int *csrColInd,
                                                             csru2csrInfo_t info, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                             double *csrVal, const int *csrRowPtr, int *csrColInd,
                                                             csru2csrInfo_t info, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                             cuComplex *csrVal, const int *csrRowPtr, int *csrColInd,
                                                             csru2csrInfo_t info, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                             cuDoubleComplex *csrVal, const int *csrRowPtr,
                                                             int *csrColInd, csru2csrInfo_t info,
                                                             size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, float *csrVal, const int *csrRowPtr,
                                               int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, double *csrVal, const int *csrRowPtr,
                                               int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, cuComplex *csrVal, const int *csrRowPtr,
                                               int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, cuDoubleComplex *csrVal,
                                               const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                                               void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, float *csrVal, const int *csrRowPtr,
                                               int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, double *csrVal, const int *csrRowPtr,
                                               int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, cuComplex *csrVal, const int *csrRowPtr,
                                               int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                               const cusparseMatDescr_t descrA, cuDoubleComplex *csrVal,
                                               const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                                               void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, const __half *threshold,
    const cusparseMatDescr_t descrC, const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda, const float *threshold,
    const cusparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, const double *threshold,
    const cusparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const __half *A, int lda,
                                                        const __half *threshold, const cusparseMatDescr_t descrC,
                                                        int *csrRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const float *A, int lda,
                                                        const float *threshold, const cusparseMatDescr_t descrC,
                                                        int *csrRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const double *A, int lda,
                                                        const double *threshold, const cusparseMatDescr_t descrC,
                                                        int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csr(cusparseHandle_t handle, int m, int n, const __half *A, int lda,
                                                     const __half *threshold, const cusparseMatDescr_t descrC,
                                                     __half *csrSortedValC, const int *csrSortedRowPtrC,
                                                     int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr(cusparseHandle_t handle, int m, int n, const float *A, int lda,
                                                     const float *threshold, const cusparseMatDescr_t descrC,
                                                     float *csrSortedValC, const int *csrSortedRowPtrC,
                                                     int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr(cusparseHandle_t handle, int m, int n, const double *A, int lda,
                                                     const double *threshold, const cusparseMatDescr_t descrC,
                                                     double *csrSortedValC, const int *csrSortedRowPtrC,
                                                     int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA,
                                                      const cusparseMatDescr_t descrA, const __half *csrSortedValA,
                                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                      const __half *threshold, const cusparseMatDescr_t descrC,
                                                      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA,
                                                      const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                      const float *threshold, const cusparseMatDescr_t descrC,
                                                      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA,
                                                      const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                      const double *threshold, const cusparseMatDescr_t descrC,
                                                      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA,
                                                   const cusparseMatDescr_t descrA, const __half *csrSortedValA,
                                                   const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                   const __half *threshold, const cusparseMatDescr_t descrC,
                                                   __half *csrSortedValC, const int *csrSortedRowPtrC,
                                                   int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA,
                                                   const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                   const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                   const float *threshold, const cusparseMatDescr_t descrC,
                                                   float *csrSortedValC, const int *csrSortedRowPtrC,
                                                   int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA,
                                                   const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                   const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                   const double *threshold, const cusparseMatDescr_t descrC,
                                                   double *csrSortedValC, const int *csrSortedRowPtrC,
                                                   int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n,
                                                                    const __half *A, int lda, float percentage,
                                                                    const cusparseMatDescr_t descrC, int *csrRowPtrC,
                                                                    int *nnzTotalDevHostPtr, pruneInfo_t info,
                                                                    void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n,
                                                                    const float *A, int lda, float percentage,
                                                                    const cusparseMatDescr_t descrC, int *csrRowPtrC,
                                                                    int *nnzTotalDevHostPtr, pruneInfo_t info,
                                                                    void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n,
                                                                    const double *A, int lda, float percentage,
                                                                    const cusparseMatDescr_t descrC, int *csrRowPtrC,
                                                                    int *nnzTotalDevHostPtr, pruneInfo_t info,
                                                                    void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const __half *A,
                                                                 int lda, float percentage,
                                                                 const cusparseMatDescr_t descrC, __half *csrSortedValC,
                                                                 const int *csrSortedRowPtrC, int *csrSortedColIndC,
                                                                 pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const float *A,
                                                                 int lda, float percentage,
                                                                 const cusparseMatDescr_t descrC, float *csrSortedValC,
                                                                 const int *csrSortedRowPtrC, int *csrSortedColIndC,
                                                                 pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const double *A,
                                                                 int lda, float percentage,
                                                                 const cusparseMatDescr_t descrC, double *csrSortedValC,
                                                                 const int *csrSortedRowPtrC, int *csrSortedColIndC,
                                                                 pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC, __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
  ava_unsupported;
}

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_MATRIX_SORTING_UNIMPLEMENTED_H_
