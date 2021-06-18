#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_PRECONDITIONER_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_PRECONDITIONER_UNIMPLEMENTED_H_
#include <cusparse.h>

#include "cava/nightwatch/parser/c/nightwatch.h"
//##############################################################################
//# PRECONDITIONERS
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseCsrilu0Ex(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                               const cusparseMatDescr_t descrA, void *csrSortedValA_ValM,
                                               cudaDataType csrSortedValA_ValMtype, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, cusparseSolveAnalysisInfo_t info,
                                               cudaDataType executiontype) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                              const cusparseMatDescr_t descrA, float *csrSortedValA_ValM,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                              const cusparseMatDescr_t descrA, double *csrSortedValA_ValM,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                              const cusparseMatDescr_t descrA, cuComplex *csrSortedValA_ValM,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                              const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA_ValM,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info,
                                                            int enable_boost, double *tol, float *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info,
                                                            int enable_boost, double *tol, double *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info,
                                                            int enable_boost, double *tol, cuComplex *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info,
                                                            int enable_boost, double *tol, cuDoubleComplex *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                          const cusparseMatDescr_t descrA, float *csrSortedValA,
                                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                          csrilu02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                          const cusparseMatDescr_t descrA, double *csrSortedValA,
                                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                          csrilu02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                          const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
                                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                          csrilu02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                          const cusparseMatDescr_t descrA,
                                                          cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA, csrilu02Info_t info,
                                                          int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                             const cusparseMatDescr_t descrA, float *csrSortedVal,
                                                             const int *csrSortedRowPtr, const int *csrSortedColInd,
                                                             csrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                             const cusparseMatDescr_t descrA, double *csrSortedVal,
                                                             const int *csrSortedRowPtr, const int *csrSortedColInd,
                                                             csrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                             const cusparseMatDescr_t descrA, cuComplex *csrSortedVal,
                                                             const int *csrSortedRowPtr, const int *csrSortedColInd,
                                                             csrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                             const cusparseMatDescr_t descrA,
                                                             cuDoubleComplex *csrSortedVal, const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd, csrilu02Info_t info,
                                                             size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                        const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                        csrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                        const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                        csrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                        const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                        csrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const cuDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                        csrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               float *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               double *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               cuComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                               cuDoubleComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info,
                                                            int enable_boost, double *tol, float *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info,
                                                            int enable_boost, double *tol, double *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info,
                                                            int enable_boost, double *tol, cuComplex *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info,
                                                            int enable_boost, double *tol, cuDoubleComplex *boost_val) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                          int nnzb, const cusparseMatDescr_t descrA,
                                                          float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                          int nnzb, const cusparseMatDescr_t descrA,
                                                          double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                          int nnzb, const cusparseMatDescr_t descrA,
                                                          cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                          int nnzb, const cusparseMatDescr_t descrA,
                                                          cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nnzb, const cusparseMatDescr_t descrA,
                                                             float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int blockSize,
                                                             bsrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nnzb, const cusparseMatDescr_t descrA,
                                                             double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int blockSize,
                                                             bsrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nnzb, const cusparseMatDescr_t descrA,
                                                             cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int blockSize,
                                                             bsrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nnzb, const cusparseMatDescr_t descrA,
                                                             cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int blockSize,
                                                             bsrilu02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                        int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal,
                                                        const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                        int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                        int nnzb, const cusparseMatDescr_t descrA, double *bsrSortedVal,
                                                        const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                        int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                        int nnzb, const cusparseMatDescr_t descrA,
                                                        cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                        int nnzb, const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                               const cusparseMatDescr_t descrA, float *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                               const cusparseMatDescr_t descrA, double *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                               const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                               const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsric0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                             const cusparseMatDescr_t descrA, float *csrSortedValA_ValM,
                                             const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                             cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                             const cusparseMatDescr_t descrA, double *csrSortedValA_ValM,
                                             const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                             cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                             const cusparseMatDescr_t descrA, cuComplex *csrSortedValA_ValM,
                                             const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                             cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric0(cusparseHandle_t handle, cusparseOperation_t trans, int m,
                                             const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA_ValM,
                                             const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                             cusparseSolveAnalysisInfo_t info) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                         const cusparseMatDescr_t descrA, float *csrSortedValA,
                                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                         csric02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                         const cusparseMatDescr_t descrA, double *csrSortedValA,
                                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                         csric02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                         const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
                                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                         csric02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                         const cusparseMatDescr_t descrA,
                                                         cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA, csric02Info_t info,
                                                         int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                            const cusparseMatDescr_t descrA, float *csrSortedVal,
                                                            const int *csrSortedRowPtr, const int *csrSortedColInd,
                                                            csric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                            const cusparseMatDescr_t descrA, double *csrSortedVal,
                                                            const int *csrSortedRowPtr, const int *csrSortedColInd,
                                                            csric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                            const cusparseMatDescr_t descrA, cuComplex *csrSortedVal,
                                                            const int *csrSortedRowPtr, const int *csrSortedColInd,
                                                            csric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz,
                                                            const cusparseMatDescr_t descrA,
                                                            cuDoubleComplex *csrSortedVal, const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd, csric02Info_t info,
                                                            size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                       const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                       csric02Info_t info, cusparseSolvePolicy_t policy,
                                                       void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                       const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                       csric02Info_t info, cusparseSolvePolicy_t policy,
                                                       void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                       const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
                                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                       csric02Info_t info, cusparseSolvePolicy_t policy,
                                                       void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                       const cusparseMatDescr_t descrA,
                                                       const cuDoubleComplex *csrSortedValA,
                                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                       csric02Info_t info, cusparseSolvePolicy_t policy,
                                                       void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                              float *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, csric02Info_t info,
                                              cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                              double *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, csric02Info_t info,
                                              cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                              cuComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, csric02Info_t info,
                                              cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                              cuDoubleComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, csric02Info_t info,
                                              cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t info, int *position) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                         int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal,
                                                         const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                         int blockDim, bsric02Info_t info, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                         int nnzb, const cusparseMatDescr_t descrA,
                                                         double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                         int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                         int nnzb, const cusparseMatDescr_t descrA,
                                                         cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                         int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                         int nnzb, const cusparseMatDescr_t descrA,
                                                         cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                         int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                            int nnzb, const cusparseMatDescr_t descrA,
                                                            float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                            const int *bsrSortedColInd, int blockSize,
                                                            bsric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                            int nnzb, const cusparseMatDescr_t descrA,
                                                            double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                            const int *bsrSortedColInd, int blockSize,
                                                            bsric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                            int nnzb, const cusparseMatDescr_t descrA,
                                                            cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                            const int *bsrSortedColInd, int blockSize,
                                                            bsric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                            int nnzb, const cusparseMatDescr_t descrA,
                                                            cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                            const int *bsrSortedColInd, int blockSize,
                                                            bsric02Info_t info, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                       int nnzb, const cusparseMatDescr_t descrA,
                                                       const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy, void *pInputBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                       int nnzb, const cusparseMatDescr_t descrA,
                                                       const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy, void *pInputBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                       int nnzb, const cusparseMatDescr_t descrA,
                                                       const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy, void *pInputBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                       int nnzb, const cusparseMatDescr_t descrA,
                                                       const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy, void *pInputBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                              const cusparseMatDescr_t descrA, float *bsrSortedVal,
                                              const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                              bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                              const cusparseMatDescr_t descrA, double *bsrSortedVal,
                                              const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                              bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                              const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
                                              const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                              bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                              const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
                                              const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
                                              bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv(cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
                                           const float *du, float *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv(cusparseHandle_t handle, int m, int n, const double *dl, const double *d,
                                           const double *du, double *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv(cusparseHandle_t handle, int m, int n, const cuComplex *dl,
                                           const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv(cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
                                           const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *B,
                                           int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float *dl,
                                                          const float *d, const float *du, const float *B, int ldb,
                                                          size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double *dl,
                                                          const double *d, const double *du, const double *B, int ldb,
                                                          size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex *dl,
                                                          const cuComplex *d, const cuComplex *du, const cuComplex *B,
                                                          int ldb, size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                          const cuDoubleComplex *dl, const cuDoubleComplex *d,
                                                          const cuDoubleComplex *du, const cuDoubleComplex *B, int ldb,
                                                          size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
                                            const float *du, float *B, int ldb, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double *dl, const double *d,
                                            const double *du, double *B, int ldb, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex *dl,
                                            const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb,
                                            void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
                                            const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *B,
                                            int ldb, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv_nopivot(cusparseHandle_t handle, int m, int n, const float *dl,
                                                   const float *d, const float *du, float *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv_nopivot(cusparseHandle_t handle, int m, int n, const double *dl,
                                                   const double *d, const double *du, double *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex *dl,
                                                   const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
                                                   const cuDoubleComplex *d, const cuDoubleComplex *du,
                                                   cuDoubleComplex *B, int ldb) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                  const float *dl, const float *d, const float *du,
                                                                  const float *B, int ldb, size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                  const double *dl, const double *d, const double *du,
                                                                  const double *B, int ldb, size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                  const cuComplex *dl, const cuComplex *d,
                                                                  const cuComplex *du, const cuComplex *B, int ldb,
                                                                  size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                  const cuDoubleComplex *dl, const cuDoubleComplex *d,
                                                                  const cuDoubleComplex *du, const cuDoubleComplex *B,
                                                                  int ldb, size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const float *dl,
                                                    const float *d, const float *du, float *B, int ldb, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const double *dl,
                                                    const double *d, const double *du, double *B, int ldb,
                                                    void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex *dl,
                                                    const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb,
                                                    void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
                                                    const cuDoubleComplex *d, const cuDoubleComplex *du,
                                                    cuDoubleComplex *B, int ldb, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsvStridedBatch(cusparseHandle_t handle, int m, const float *dl, const float *d,
                                                       const float *du, float *x, int batchCount, int batchStride) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsvStridedBatch(cusparseHandle_t handle, int m, const double *dl,
                                                       const double *d, const double *du, double *x, int batchCount,
                                                       int batchStride) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsvStridedBatch(cusparseHandle_t handle, int m, const cuComplex *dl,
                                                       const cuComplex *d, const cuComplex *du, cuComplex *x,
                                                       int batchCount, int batchStride) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsvStridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
                                                       const cuDoubleComplex *d, const cuDoubleComplex *du,
                                                       cuDoubleComplex *x, int batchCount, int batchStride) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float *dl,
                                                                      const float *d, const float *du, const float *x,
                                                                      int batchCount, int batchStride,
                                                                      size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double *dl,
                                                                      const double *d, const double *du,
                                                                      const double *x, int batchCount, int batchStride,
                                                                      size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m,
                                                                      const cuComplex *dl, const cuComplex *d,
                                                                      const cuComplex *du, const cuComplex *x,
                                                                      int batchCount, int batchStride,
                                                                      size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *x, int batchCount, int batchStride, size_t *bufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float *dl, const float *d,
                                                        const float *du, float *x, int batchCount, int batchStride,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double *dl,
                                                        const double *d, const double *du, double *x, int batchCount,
                                                        int batchStride, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuComplex *dl,
                                                        const cuComplex *d, const cuComplex *du, cuComplex *x,
                                                        int batchCount, int batchStride, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
                                                        const cuDoubleComplex *d, const cuDoubleComplex *du,
                                                        cuDoubleComplex *x, int batchCount, int batchStride,
                                                        void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m,
                                                                         const float *dl, const float *d,
                                                                         const float *du, const float *x,
                                                                         int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m,
                                                                         const double *dl, const double *d,
                                                                         const double *du, const double *x,
                                                                         int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m,
                                                                         const cuComplex *dl, const cuComplex *d,
                                                                         const cuComplex *du, const cuComplex *x,
                                                                         int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d,
    const cuDoubleComplex *du, const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float *dl,
                                                           float *d, float *du, float *x, int batchCount,
                                                           void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double *dl,
                                                           double *d, double *du, double *x, int batchCount,
                                                           void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex *dl,
                                                           cuComplex *d, cuComplex *du, cuComplex *x, int batchCount,
                                                           void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                           cuDoubleComplex *dl, cuDoubleComplex *d, cuDoubleComplex *du,
                                                           cuDoubleComplex *x, int batchCount, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m,
                                                                         const float *ds, const float *dl,
                                                                         const float *d, const float *du,
                                                                         const float *dw, const float *x,
                                                                         int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m,
                                                                         const double *ds, const double *dl,
                                                                         const double *d, const double *du,
                                                                         const double *dw, const double *x,
                                                                         int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m,
                                                                         const cuComplex *ds, const cuComplex *dl,
                                                                         const cuComplex *d, const cuComplex *du,
                                                                         const cuComplex *dw, const cuComplex *x,
                                                                         int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *ds, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *dw, const cuDoubleComplex *x,
    int batchCount, size_t *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float *ds,
                                                           float *dl, float *d, float *du, float *dw, float *x,
                                                           int batchCount, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double *ds,
                                                           double *dl, double *d, double *du, double *dw, double *x,
                                                           int batchCount, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex *ds,
                                                           cuComplex *dl, cuComplex *d, cuComplex *du, cuComplex *dw,
                                                           cuComplex *x, int batchCount, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                           cuDoubleComplex *ds, cuDoubleComplex *dl, cuDoubleComplex *d,
                                                           cuDoubleComplex *du, cuDoubleComplex *dw, cuDoubleComplex *x,
                                                           int batchCount, void *pBuffer) {
  ava_unsupported;
}
#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_PRECONDITIONER_UNIMPLEMENTED_H_
