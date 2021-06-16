#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL2_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL2_UNIMPLEMENTED_H_
#include <cusparse.h>
//##############################################################################
//# SPARSE LEVEL 2 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
                                            const float *alpha, const float *A, int lda, int nnz, const float *xVal,
                                            const int *xInd, const float *beta, float *y, cusparseIndexBase_t idxBase,
                                            void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                       int n, int nnz, int *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
                                            const double *alpha, const double *A, int lda, int nnz, const double *xVal,
                                            const int *xInd, const double *beta, double *y, cusparseIndexBase_t idxBase,
                                            void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                       int n, int nnz, int *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
                                            const cuComplex *alpha, const cuComplex *A, int lda, int nnz,
                                            const cuComplex *xVal, const int *xInd, const cuComplex *beta, cuComplex *y,
                                            cusparseIndexBase_t idxBase, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                       int n, int nnz, int *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
                                            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, int nnz,
                                            const cuDoubleComplex *xVal, const int *xInd, const cuDoubleComplex *beta,
                                            cuDoubleComplex *y, cusparseIndexBase_t idxBase, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                       int n, int nnz, int *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx_bufferSize(cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA, int m, int n,
                           int nnz, const void *alpha, cudaDataType alphatype, const cusparseMatDescr_t descrA,
                           const void *csrValA, cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
                           const void *x, cudaDataType xtype, const void *beta, cudaDataType betatype, void *y,
                           cudaDataType ytype, cudaDataType executiontype, size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx(cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
                                             int m, int n, int nnz, const void *alpha, cudaDataType alphatype,
                                             const cusparseMatDescr_t descrA, const void *csrValA,
                                             cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
                                             const void *x, cudaDataType xtype, const void *beta, cudaDataType betatype,
                                             void *y, cudaDataType ytype, cudaDataType executiontype, void *buffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseShybmv(cusparseHandle_t handle, cusparseOperation_t transA, const float *alpha,
                                            const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA,
                                            const float *x, const float *beta, float *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDhybmv(cusparseHandle_t handle, cusparseOperation_t transA, const double *alpha,
                                            const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA,
                                            const double *x, const double *beta, double *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseChybmv(cusparseHandle_t handle, cusparseOperation_t transA, const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA,
                                            const cuComplex *x, const cuComplex *beta, cuComplex *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZhybmv(cusparseHandle_t handle, cusparseOperation_t transA,
                                            const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA, const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta, cuDoubleComplex *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, int mb, int nb, int nnzb, const float *alpha,
                                            const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                            const float *x, const float *beta, float *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, int mb, int nb, int nnzb, const double *alpha,
                                            const cusparseMatDescr_t descrA, const double *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                            const double *x, const double *beta, double *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, int mb, int nb, int nnzb,
                                            const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                            const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA, int blockDim, const cuComplex *x,
                                            const cuComplex *beta, cuComplex *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                            cusparseOperation_t transA, int mb, int nb, int nnzb,
                                            const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta, cuDoubleComplex *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                             cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
                                             const float *alpha, const cusparseMatDescr_t descrA,
                                             const float *bsrSortedValA, const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA, int blockDim, const float *x,
                                             const float *beta, float *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                             cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
                                             const double *alpha, const cusparseMatDescr_t descrA,
                                             const double *bsrSortedValA, const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA, int blockDim, const double *x,
                                             const double *beta, double *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                             cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
                                             const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                             const cuComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA, int blockDim, const cuComplex *x,
                                             const cuComplex *beta, cuComplex *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                             cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
                                             const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                             const cuDoubleComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x,
                                             const cuDoubleComplex *beta, cuDoubleComplex *y) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle, csrsv2Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                        int nnz, const cusparseMatDescr_t descrA, float *csrSortedValA,
                                                        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                        csrsv2Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                        int nnz, const cusparseMatDescr_t descrA, double *csrSortedValA,
                                                        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                        csrsv2Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                        int nnz, const cusparseMatDescr_t descrA,
                                                        cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA, csrsv2Info_t info,
                                                        int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                        int nnz, const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA, csrsv2Info_t info,
                                                        int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                           int nnz, const cusparseMatDescr_t descrA,
                                                           float *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, csrsv2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                           int nnz, const cusparseMatDescr_t descrA,
                                                           double *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, csrsv2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                           int nnz, const cusparseMatDescr_t descrA,
                                                           cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, csrsv2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                           int nnz, const cusparseMatDescr_t descrA,
                                                           cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, csrsv2Info_t info,
                                                           size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                      int nnz, const cusparseMatDescr_t descrA,
                                                      const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                      int nnz, const cusparseMatDescr_t descrA,
                                                      const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                      int nnz, const cusparseMatDescr_t descrA,
                                                      const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                      int nnz, const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA, csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                                                   const float *alpha, const cusparseMatDescr_t descrA,
                                                   const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, csrsv2Info_t info, const float *f,
                                                   float *x, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                                                   const double *alpha, const cusparseMatDescr_t descrA,
                                                   const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, csrsv2Info_t info, const double *f,
                                                   double *x, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                                                   const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                                   const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, csrsv2Info_t info, const cuComplex *f,
                                                   cuComplex *x, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                                                   const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                                   const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, csrsv2Info_t info,
                                                   const cuDoubleComplex *f, cuDoubleComplex *x,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, int mb, int nnzb,
                                                        const cusparseMatDescr_t descrA, float *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                        int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, int mb, int nnzb,
                                                        const cusparseMatDescr_t descrA, double *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                        int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, int mb, int nnzb,
                                                        const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                        int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                        cusparseOperation_t transA, int mb, int nnzb,
                                                        const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                        int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, int mb, int nnzb,
                                                           const cusparseMatDescr_t descrA, float *bsrSortedValA,
                                                           const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                           int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, int mb, int nnzb,
                                                           const cusparseMatDescr_t descrA, double *bsrSortedValA,
                                                           const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                           int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, int mb, int nnzb,
                                                           const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA,
                                                           const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                           int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                           cusparseOperation_t transA, int mb, int nnzb,
                                                           const cusparseMatDescr_t descrA,
                                                           cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                           const int *bsrSortedColIndA, int blockSize,
                                                           bsrsv2Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, int mb, int nnzb,
                                                      const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                      int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
                                                      void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, int mb, int nnzb,
                                                      const cusparseMatDescr_t descrA, const double *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                      int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
                                                      void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, int mb, int nnzb,
                                                      const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                      int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
                                                      void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                      cusparseOperation_t transA, int mb, int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, int mb, int nnzb, const float *alpha,
                                                   const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                   int blockDim, bsrsv2Info_t info, const float *f, float *x,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, int mb, int nnzb, const double *alpha,
                                                   const cusparseMatDescr_t descrA, const double *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                   int blockDim, bsrsv2Info_t info, const double *f, double *x,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, int mb, int nnzb, const cuComplex *alpha,
                                                   const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                   int blockDim, bsrsv2Info_t info, const cuComplex *f, cuComplex *x,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                   cusparseOperation_t transA, int mb, int nnzb,
                                                   const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                                   const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info,
                                                   const cuDoubleComplex *f, cuDoubleComplex *x,
                                                   cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseShybsv_analysis(cusparseHandle_t handle, cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA, cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDhybsv_analysis(cusparseHandle_t handle, cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA, cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseChybsv_analysis(cusparseHandle_t handle, cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA, cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZhybsv_analysis(cusparseHandle_t handle, cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA, cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseShybsv_solve(cusparseHandle_t handle, cusparseOperation_t trans,
                                                  const float *alpha, const cusparseMatDescr_t descrA,
                                                  const cusparseHybMat_t hybA, cusparseSolveAnalysisInfo_t info,
                                                  const float *f, float *x) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseChybsv_solve(cusparseHandle_t handle, cusparseOperation_t trans,
                                                  const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                                  const cusparseHybMat_t hybA, cusparseSolveAnalysisInfo_t info,
                                                  const cuComplex *f, cuComplex *x) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDhybsv_solve(cusparseHandle_t handle, cusparseOperation_t trans,
                                                  const double *alpha, const cusparseMatDescr_t descrA,
                                                  const cusparseHybMat_t hybA, cusparseSolveAnalysisInfo_t info,
                                                  const double *f, double *x) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZhybsv_solve(cusparseHandle_t handle, cusparseOperation_t trans,
                                                  const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                                                  const cusparseHybMat_t hybA, cusparseSolveAnalysisInfo_t info,
                                                  const cuDoubleComplex *f, cuDoubleComplex *x) {
  ava_unsupported;
}

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL2_UNIMPLEMENTED_H_
