#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL1_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL1_UNIMPLEMENTED_H_
#include <cusparse.h>

#include "cava/nightwatch/parser/c/nightwatch.h"
//##############################################################################
//# SPARSE LEVEL 1 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float *alpha, const float *xVal,
                                            const int *xInd, float *y, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double *alpha, const double *xVal,
                                            const int *xInd, double *y, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, int nnz, const cuComplex *alpha,
                                            const cuComplex *xVal, const int *xInd, cuComplex *y,
                                            cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, int nnz, const cuDoubleComplex *alpha,
                                            const cuDoubleComplex *xVal, const int *xInd, cuDoubleComplex *y,
                                            cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, int nnz, const float *y, float *xVal,
                                           const int *xInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, int nnz, const double *y, double *xVal,
                                           const int *xInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, int nnz, const cuComplex *y, cuComplex *xVal,
                                           const int *xInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, int nnz, const cuDoubleComplex *y,
                                           cuDoubleComplex *xVal, const int *xInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, int nnz, float *y, float *xVal, const int *xInd,
                                            cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, int nnz, double *y, double *xVal, const int *xInd,
                                            cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, int nnz, cuComplex *y, cuComplex *xVal,
                                            const int *xInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, int nnz, cuDoubleComplex *y, cuDoubleComplex *xVal,
                                            const int *xInd, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, int nnz, const float *xVal, const int *xInd,
                                           float *y, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, int nnz, const double *xVal, const int *xInd,
                                           double *y, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, int nnz, const cuComplex *xVal, const int *xInd,
                                           cuComplex *y, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, int nnz, const cuDoubleComplex *xVal,
                                           const int *xInd, cuDoubleComplex *y, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, int nnz, float *xVal, const int *xInd, float *y,
                                           const float *c, const float *s, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, int nnz, double *xVal, const int *xInd, double *y,
                                           const double *c, const double *s, cusparseIndexBase_t idxBase) {
  ava_unsupported;
}
#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_SPARSE_LEVEL1_UNIMPLEMENTED_H_
