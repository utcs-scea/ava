#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_CSR2CSC_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_CSR2CSC_UNIMPLEMENTED_H_
#include <cusparse.h>
//##############################################################################
//# CSR2CSC
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
                                                const int *csrRowPtr, const int *csrColInd, void *cscVal,
                                                int *cscColPtr, int *cscRowInd, cudaDataType valType,
                                                cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                                cusparseCsr2CscAlg_t alg, void *buffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz,
                                                           const void *csrVal, const int *csrRowPtr,
                                                           const int *csrColInd, void *cscVal, int *cscColPtr,
                                                           int *cscRowInd, cudaDataType valType,
                                                           cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                                           cusparseCsr2CscAlg_t alg, size_t *bufferSize) {
  ava_unsupported;
}

//------------------------------------------------------------------------------
// SPARSE VECTOR DESCRIPTOR

cusparseStatus_t CUSPARSEAPI cusparseCreateSpVec(cusparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz,
                                                 void *indices, void *values, cusparseIndexType_t idxType,
                                                 cusparseIndexBase_t idxBase, cudaDataType valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(const cusparseSpVecDescr_t spVecDescr, int64_t *size, int64_t *nnz,
                                              void **indices, void **values, cusparseIndexType_t *idxType,
                                              cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(const cusparseSpVecDescr_t spVecDescr,
                                                       cusparseIndexBase_t *idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpVecGetValues(const cusparseSpVecDescr_t spVecDescr, void **values) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void *values) { ava_unsupported; }

//------------------------------------------------------------------------------
// DENSE VECTOR DESCRIPTOR

cusparseStatus_t CUSPARSEAPI cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size, void *values,
                                                 cudaDataType valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDnVecGet(const cusparseDnVecDescr_t dnVecDescr, int64_t *size, void **values,
                                              cudaDataType *valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnVecGetValues(const cusparseDnVecDescr_t dnVecDescr, void **values) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void *values) { ava_unsupported; }

//------------------------------------------------------------------------------
// SPARSE MATRIX DESCRIPTOR

cusparseStatus_t CUSPARSEAPI cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
                                               int64_t nnz, void *cooRowInd, void *cooColInd, void *cooValues,
                                               cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase,
                                               cudaDataType valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
                                               int64_t nnz, void *csrRowOffsets, void *csrColInd, void *csrValues,
                                               cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType,
                                               cusparseIndexBase_t idxBase, cudaDataType valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCooAoS(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
                                                  int64_t nnz, void *cooInd, void *cooValues,
                                                  cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase,
                                                  cudaDataType valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCooGet(const cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
                                            int64_t *nnz,
                                            void **cooRowInd,  // COO row indices
                                            void **cooColInd,  // COO column indices
                                            void **cooValues,  // COO values
                                            cusparseIndexType_t *idxType, cusparseIndexBase_t *idxBase,
                                            cudaDataType *valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCooAoSGet(const cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
                                               int64_t *nnz,
                                               void **cooInd,     // COO indices
                                               void **cooValues,  // COO values
                                               cusparseIndexType_t *idxType, cusparseIndexBase_t *idxBase,
                                               cudaDataType *valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCsrGet(const cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
                                            int64_t *nnz, void **csrRowOffsets, void **csrColInd, void **csrValues,
                                            cusparseIndexType_t *csrRowOffsetsType, cusparseIndexType_t *csrColIndType,
                                            cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(const cusparseSpMatDescr_t spMatDescr, cusparseFormat_t *format) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(const cusparseSpMatDescr_t spMatDescr,
                                                       cusparseIndexBase_t *idxBase) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetValues(const cusparseSpMatDescr_t spMatDescr, void **values) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void *values) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(const cusparseSpMatDescr_t spMatDescr, int *batchCount) {
  ava_unsupported;
}

//------------------------------------------------------------------------------
// DENSE MATRIX DESCRIPTOR

cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols,
                                                 int64_t ld, void *values, cudaDataType valueType,
                                                 cusparseOrder_t order) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(const cusparseDnMatDescr_t dnMatDescr, int64_t *rows, int64_t *cols,
                                              int64_t *ld, void **values, cudaDataType *type, cusparseOrder_t *order) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnMatGetValues(const cusparseDnMatDescr_t dnMatDescr, void **values) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void *values) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount,
                                                          int64_t batchStride) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(const cusparseDnMatDescr_t dnMatDescr, int *batchCount,
                                                          int64_t *batchStride) {
  ava_unsupported;
}

//------------------------------------------------------------------------------
// SPARSE VECTOR-VECTOR MULTIPLICATION

cusparseStatus_t CUSPARSEAPI cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX,
                                          const cusparseSpVecDescr_t vecX, const cusparseDnVecDescr_t vecY,
                                          void *result, cudaDataType computeType, void *externalBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX,
                                                     const cusparseSpVecDescr_t vecX, const cusparseDnVecDescr_t vecY,
                                                     const void *result, cudaDataType computeType, size_t *bufferSize) {
  ava_unsupported;
}

//------------------------------------------------------------------------------
// SPARSE MATRIX-VECTOR MULTIPLICATION

cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
                                          const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
                                          const void *beta, const cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                          cusparseSpMVAlg_t alg, void *externalBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                                     const void *alpha, const cusparseSpMatDescr_t matA,
                                                     const cusparseDnVecDescr_t vecX, const void *beta,
                                                     const cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                                     cusparseSpMVAlg_t alg, size_t *bufferSize) {
  ava_unsupported;
}

//------------------------------------------------------------------------------
// SPARSE MATRIX-MATRIX MULTIPLICATION

cusparseStatus_t CUSPARSEAPI cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                          const void *alpha, const cusparseSpMatDescr_t matA,
                                          const cusparseDnMatDescr_t matB, const void *beta, cusparseDnMatDescr_t matC,
                                          cudaDataType computeType, cusparseSpMMAlg_t alg, void *externalBuffer) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                                     cusparseOperation_t opB, const void *alpha,
                                                     const cusparseSpMatDescr_t matA, const cusparseDnMatDescr_t matB,
                                                     const void *beta, cusparseDnMatDescr_t matC,
                                                     cudaDataType computeType, cusparseSpMMAlg_t alg,
                                                     size_t *bufferSize) {
  ava_unsupported;
}

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_CSR2CSC_UNIMPLEMENTED_H_
