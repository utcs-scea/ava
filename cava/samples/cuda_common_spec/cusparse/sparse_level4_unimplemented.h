#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL4_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL4_UNIMPLEMENTED_H_
#include <cusparse.h>

#include "cava/nightwatch/parser/c/nightwatch.h"
//##############################################################################
//# SPARSE LEVEL 4 ROUTINES #
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrgemm2Info(csrgemm2Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrgemm2Info(csrgemm2Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const float *beta, const cusparseMatDescr_t descrD,
    int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const double *beta, const cusparseMatDescr_t descrD,
    int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cuComplex *beta, const cusparseMatDescr_t descrD,
    int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrD, int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Nnz(
    cusparseHandle_t handle, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD, const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, const csrgemm2Info_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2(cusparseHandle_t handle, int m, int n, int k, const float *alpha,
                                               const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
                                               const int *csrSortedRowPtrB, const int *csrSortedColIndB,
                                               const float *beta, const cusparseMatDescr_t descrD, int nnzD,
                                               const float *csrSortedValD, const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
                                               float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
                                               const csrgemm2Info_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const double *alpha,
                                               const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB,
                                               const int *csrSortedRowPtrB, const int *csrSortedColIndB,
                                               const double *beta, const cusparseMatDescr_t descrD, int nnzD,
                                               const double *csrSortedValD, const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
                                               double *csrSortedValC, const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC, const csrgemm2Info_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cuComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const cuComplex *csrSortedValD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, cuComplex *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    int nnzA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cuDoubleComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const cuDoubleComplex *csrSortedValD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *beta,
    const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const double *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *beta,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const cuComplex *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuDoubleComplex *beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n,
                                                  const cusparseMatDescr_t descrA, int nnzA,
                                                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                  const cusparseMatDescr_t descrB, int nnzB,
                                                  const int *csrSortedRowPtrB, const int *csrSortedColIndB,
                                                  const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
                                                  int *nnzTotalDevHostPtr, void *workspace) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float *alpha,
                                               const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               const float *beta, const cusparseMatDescr_t descrB, int nnzB,
                                               const float *csrSortedValB, const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                                               float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC,
                                               void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double *alpha,
                                               const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               const double *beta, const cusparseMatDescr_t descrB, int nnzB,
                                               const double *csrSortedValB, const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                                               double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC,
                                               void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex *alpha, const cusparseMatDescr_t descrA,
                  int nnzA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const cuComplex *beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB,
                  const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                  cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuDoubleComplex *beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
  ava_unsupported;
}

/* --- Sparse Matrix Reorderings --- */

/* Description: Find an approximate coloring of a matrix stored in CSR format.
 */
cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               const float *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const float *fractionToColor, int *ncolors,
                                               int *coloring, int *reordering, const cusparseColorInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               const double *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
                                               int *coloring, int *reordering, const cusparseColorInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const float *fractionToColor, int *ncolors,
                                               int *coloring, int *reordering, const cusparseColorInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
                                               int *coloring, int *reordering, const cusparseColorInfo_t info) {
  ava_unsupported;
}

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL4_UNIMPLEMENTED_H_
