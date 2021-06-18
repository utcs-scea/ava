#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_UNIMPLEMENTED_H_
#include <cusparse.h>

#include "cava/nightwatch/parser/c/nightwatch.h"
//##############################################################################
//# INITILIAZATION AND MANAGMENT ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int *version) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseGetProperty(libraryPropertyType type, int *value) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  ava_unsupported;
}

//##############################################################################
//# HELPER ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t *descrA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr(cusparseMatDescr_t descrA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
  ava_unsupported;
}

cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
  ava_unsupported;
}

cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) {
  ava_unsupported;
}

cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base_) {
  ava_unsupported;
}

cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseGetLevelInfo(cusparseHandle_t handle, cusparseSolveAnalysisInfo_t info,
                                                  int *nlevels, int **levelPtr, int **levelInd) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsv2Info(csrsv2Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsv2Info(csrsv2Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateCsric02Info(csric02Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsric02Info(csric02Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateBsric02Info(bsric02Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsric02Info(bsric02Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02Info(csrilu02Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02Info(csrilu02Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrilu02Info(bsrilu02Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrilu02Info(bsrilu02Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsv2Info(bsrsv2Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsv2Info(bsrsv2Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsm2Info(bsrsm2Info_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsm2Info(bsrsm2Info_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateHybMat(cusparseHybMat_t *hybA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyHybMat(cusparseHybMat_t hybA) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateCsru2csrInfo(csru2csrInfo_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsru2csrInfo(csru2csrInfo_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseCreateColorInfo(cusparseColorInfo_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyColorInfo(cusparseColorInfo_t info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseSetColorAlgs(cusparseColorInfo_t info, cusparseColorAlg_t alg) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseGetColorAlgs(cusparseColorInfo_t info, cusparseColorAlg_t *alg) {
  ava_unsupported;
}

cusparseStatus_t CUSPARSEAPI cusparseCreatePruneInfo(pruneInfo_t *info) { ava_unsupported; }

cusparseStatus_t CUSPARSEAPI cusparseDestroyPruneInfo(pruneInfo_t info) { ava_unsupported; }

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_UNIMPLEMENTED_H_
