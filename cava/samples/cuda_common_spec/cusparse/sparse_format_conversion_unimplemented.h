#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_FORMAT_CONVERSION_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_FORMAT_CONVERSION_UNIMPLEMENTED_H_
#include <cusparse.h>
//##############################################################################
//# SPARSE FORMAT CONVERSION
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                          const cusparseMatDescr_t descrA, const float *A, int lda, int *nnzPerRowCol,
                                          int *nnzTotalDevHostPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                          const cusparseMatDescr_t descrA, const double *A, int lda, int *nnzPerRowCol,
                                          int *nnzTotalDevHostPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                          const cusparseMatDescr_t descrA, const cuComplex *A, int lda,
                                          int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                          const cusparseMatDescr_t descrA, const cuDoubleComplex *A, int lda,
                                          int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  ava_unsupported;
}

//##############################################################################
//# SPARSE FORMAT CONVERSION #
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseSnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
                                                   const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                   int *nnzPerRow, int *nnzC, float tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
                                                   const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                   int *nnzPerRow, int *nnzC, double tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
                                                   const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                   int *nnzPerRow, int *nnzC, cuComplex tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
                                                   const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                   int *nnzPerRow, int *nnzC, cuDoubleComplex tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2csr_compress(cusparseHandle_t handle, int m, int n,
                                                       const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                       const int *csrSortedColIndA, const int *csrSortedRowPtrA,
                                                       int nnzA, const int *nnzPerRow, float *csrSortedValC,
                                                       int *csrSortedColIndC, int *csrSortedRowPtrC, float tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csr_compress(cusparseHandle_t handle, int m, int n,
                                                       const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                       const int *csrSortedColIndA, const int *csrSortedRowPtrA,
                                                       int nnzA, const int *nnzPerRow, double *csrSortedValC,
                                                       int *csrSortedColIndC, int *csrSortedRowPtrC, double tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csr_compress(cusparseHandle_t handle, int m, int n,
                                                       const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
                                                       const int *csrSortedColIndA, const int *csrSortedRowPtrA,
                                                       int nnzA, const int *nnzPerRow, cuComplex *csrSortedValC,
                                                       int *csrSortedColIndC, int *csrSortedRowPtrC, cuComplex tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    cuDoubleComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC, cuDoubleComplex tol) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const float *A, int lda, const int *nnzPerRow, float *csrSortedValA,
                                                int *csrSortedRowPtrA, int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const double *A, int lda, const int *nnzPerRow, double *csrSortedValA,
                                                int *csrSortedRowPtrA, int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuComplex *A, int lda, const int *nnzPerRow,
                                                cuComplex *csrSortedValA, int *csrSortedRowPtrA,
                                                int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A, int lda, const int *nnzPerRow,
                                                cuDoubleComplex *csrSortedValA, int *csrSortedRowPtrA,
                                                int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA, float *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA, double *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA, cuComplex *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const float *A, int lda, const int *nnzPerCol, float *cscSortedValA,
                                                int *cscSortedRowIndA, int *cscSortedColPtrA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const double *A, int lda, const int *nnzPerCol, double *cscSortedValA,
                                                int *cscSortedRowIndA, int *cscSortedColPtrA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuComplex *A, int lda, const int *nnzPerCol,
                                                cuComplex *cscSortedValA, int *cscSortedRowIndA,
                                                int *cscSortedColPtrA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A, int lda, const int *nnzPerCol,
                                                cuDoubleComplex *cscSortedValA, int *cscSortedRowIndA,
                                                int *cscSortedColPtrA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const float *cscSortedValA, const int *cscSortedRowIndA,
                                                const int *cscSortedColPtrA, float *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const double *cscSortedValA, const int *cscSortedRowIndA,
                                                const int *cscSortedColPtrA, double *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuComplex *cscSortedValA, const int *cscSortedRowIndA,
                                                const int *cscSortedColPtrA, cuComplex *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *cscSortedValA, const int *cscSortedRowIndA,
                                                const int *cscSortedColPtrA, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle, const int *cooRowInd, int nnz, int m,
                                              int *csrSortedRowPtr, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle, const int *csrSortedRowPtr, int nnz, int m,
                                              int *cooRowInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSdense2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const float *A, int lda, const int *nnzPerRow, cusparseHybMat_t hybA,
                                                int userEllWidth, cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDdense2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const double *A, int lda, const int *nnzPerRow, cusparseHybMat_t hybA,
                                                int userEllWidth, cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCdense2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuComplex *A, int lda, const int *nnzPerRow,
                                                cusparseHybMat_t hybA, int userEllWidth,
                                                cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZdense2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A, int lda, const int *nnzPerRow,
                                                cusparseHybMat_t hybA, int userEllWidth,
                                                cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseShyb2dense(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA, float *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDhyb2dense(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA, double *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseChyb2dense(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA, cuComplex *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZhyb2dense(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA, cuDoubleComplex *A, int lda) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const float *csrSortedValA, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const double *csrSortedValA, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseShyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, float *csrSortedValA, int *csrSortedRowPtrA,
                                              int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDhyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, double *csrSortedValA, int *csrSortedRowPtrA,
                                              int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseChyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, cuComplex *csrSortedValA,
                                              int *csrSortedRowPtrA, int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZhyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, cuDoubleComplex *csrSortedValA,
                                              int *csrSortedRowPtrA, int *csrSortedColIndA) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsc2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const float *cscSortedValA, const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsc2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const double *cscSortedValA, const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsc2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const cuComplex *cscSortedValA, const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsc2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *cscSortedValA, const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA, cusparseHybMat_t hybA, int userEllWidth,
                                              cusparseHybPartition_t partitionType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseShyb2csc(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, float *cscSortedVal, int *cscSortedRowInd,
                                              int *cscSortedColPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDhyb2csc(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, double *cscSortedVal, int *cscSortedRowInd,
                                              int *cscSortedColPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseChyb2csc(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, cuComplex *cscSortedVal,
                                              int *cscSortedRowInd, int *cscSortedColPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZhyb2csc(cusparseHandle_t handle, const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA, cuDoubleComplex *cscSortedVal,
                                              int *cscSortedRowInd, int *cscSortedColPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                                 const cusparseMatDescr_t descrA, const int *csrSortedRowPtrA,
                                                 const int *csrSortedColIndA, int blockDim,
                                                 const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC,
                                                 int *nnzTotalDevHostPtr) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                              const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, float *bsrSortedValC,
                                              int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                              const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, double *bsrSortedValC,
                                              int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                              const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
                                              int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                              const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
                                              int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                              const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                              const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, float *csrSortedValC,
                                              int *csrSortedRowPtrC, int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                              const cusparseMatDescr_t descrA, const double *bsrSortedValA,
                                              const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, double *csrSortedValC,
                                              int *csrSortedRowPtrC, int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                              const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
                                              const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, cuComplex *csrSortedValC,
                                              int *csrSortedRowPtrC, int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                              const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
                                              const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                              const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC,
                                              int *csrSortedRowPtrC, int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                             const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int rowBlockDim,
                                                             int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                             const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int rowBlockDim,
                                                             int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                             const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd, int rowBlockDim,
                                                             int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                             const cuDoubleComplex *bsrSortedVal,
                                                             const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                             int rowBlockDim, int colBlockDim,
                                                             int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                                const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd, int rowBlockDim,
                                                                int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                                const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd, int rowBlockDim,
                                                                int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                                const cuComplex *bsrSortedVal,
                                                                const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                                int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                                const cuDoubleComplex *bsrSortedVal,
                                                                const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                                int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                  const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                  float *bscVal, int *bscRowInd, int *bscColPtr,
                                                  cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                                  void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                  const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                  double *bscVal, int *bscRowInd, int *bscColPtr,
                                                  cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                                  void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                  const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                  cuComplex *bscVal, int *bscRowInd, int *bscColPtr,
                                                  cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                                  void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                  const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                  cuDoubleComplex *bscVal, int *bscRowInd, int *bscColPtr,
                                                  cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                                  void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                const cusparseMatDescr_t descrA, const int *bsrSortedRowPtrA,
                                                const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim,
                                                const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
                                                int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                                const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC,
                                                float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                const cusparseMatDescr_t descrA, const double *bsrSortedValA,
                                                const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC,
                                                double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
                                                const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC,
                                                cuComplex *csrSortedValC, int *csrSortedRowPtrC,
                                                int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
                                                const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC,
                                                cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
                                                int *csrSortedColIndC) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                           int n, const cusparseMatDescr_t descrA,
                                                           const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, int rowBlockDim,
                                                           int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                           int n, const cusparseMatDescr_t descrA,
                                                           const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, int rowBlockDim,
                                                           int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                           int n, const cusparseMatDescr_t descrA,
                                                           const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA, int rowBlockDim,
                                                           int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                           int n, const cusparseMatDescr_t descrA,
                                                           const cuDoubleComplex *csrSortedValA,
                                                           const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                           int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                              int n, const cusparseMatDescr_t descrA,
                                                              const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA, int rowBlockDim,
                                                              int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                              int n, const cusparseMatDescr_t descrA,
                                                              const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA, int rowBlockDim,
                                                              int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                              int n, const cusparseMatDescr_t descrA,
                                                              const cuComplex *csrSortedValA,
                                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                              int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                              int n, const cusparseMatDescr_t descrA,
                                                              const cuDoubleComplex *csrSortedValA,
                                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                              int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                                   const cusparseMatDescr_t descrA, const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA, const cusparseMatDescr_t descrC,
                                                   int *bsrSortedRowPtrC, int rowBlockDim, int colBlockDim,
                                                   int *nnzTotalDevHostPtr, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                                const cusparseMatDescr_t descrA, const float *csrSortedValA,
                                                const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC, float *bsrSortedValC,
                                                int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
                                                int colBlockDim, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                                const cusparseMatDescr_t descrA, const double *csrSortedValA,
                                                const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC, double *bsrSortedValC,
                                                int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
                                                int colBlockDim, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                                const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
                                                const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
                                                int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
                                                int colBlockDim, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
                                                const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
                                                const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
                                                int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
                                                int colBlockDim, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nb, int nnzb, const cusparseMatDescr_t descrA,
                                                             const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA, int rowBlockDimA,
                                                             int colBlockDimA, int rowBlockDimC, int colBlockDimC,
                                                             int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nb, int nnzb, const cusparseMatDescr_t descrA,
                                                             const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA, int rowBlockDimA,
                                                             int colBlockDimA, int rowBlockDimC, int colBlockDimC,
                                                             int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nb, int nnzb, const cusparseMatDescr_t descrA,
                                                             const cuComplex *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                             int rowBlockDimA, int colBlockDimA, int rowBlockDimC,
                                                             int colBlockDimC, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                             int nb, int nnzb, const cusparseMatDescr_t descrA,
                                                             const cuDoubleComplex *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                             int rowBlockDimA, int colBlockDimA, int rowBlockDimC,
                                                             int colBlockDimC, int *pBufferSizeInBytes) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                     int nnzb, const cusparseMatDescr_t descrA,
                                                     const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                     int rowBlockDimA, int colBlockDimA,
                                                     const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC,
                                                     int rowBlockDimC, int colBlockDimC, int *nnzTotalDevHostPtr,
                                                     void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                  int nnzb, const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                  int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC,
                                                  float *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC,
                                                  int rowBlockDimC, int colBlockDimC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                  int nnzb, const cusparseMatDescr_t descrA,
                                                  const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                  const cusparseMatDescr_t descrC, double *bsrSortedValC,
                                                  int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
                                                  int colBlockDimC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                  int nnzb, const cusparseMatDescr_t descrA,
                                                  const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                  const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
                                                  int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
                                                  int colBlockDimC, void *pBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                                                  int nnzb, const cusparseMatDescr_t descrA,
                                                  const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                  const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
                                                  int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
                                                  int colBlockDimC, void *pBuffer) {
  ava_unsupported;
}

#endif // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_FORMAT_CONVERSION_UNIMPLEMENTED_H_
