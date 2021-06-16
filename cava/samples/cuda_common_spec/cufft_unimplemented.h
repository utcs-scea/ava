#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUFFT_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUFFT_UNIMPLEMENTED_H_
#include <cufft.h>

/******** cufft *********/
cufftResult CUFFTAPI cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) { ava_unsupported; }

cufftResult CUFFTAPI cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) { ava_unsupported; }

cufftResult CUFFTAPI cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) { ava_unsupported; }

cufftResult CUFFTAPI cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist, cufftType type, int batch) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist, cufftType type, int batch,
                                       size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                                         long long int istride, long long int idist, long long int *onembed,
                                         long long int ostride, long long int odist, cufftType type,
                                         long long int batch, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                                        long long int istride, long long int idist, long long int *onembed,
                                        long long int ostride, long long int odist, cufftType type, long long int batch,
                                        size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftEstimate1d(int nx, cufftType type, int batch, size_t *workSize) { ava_unsupported; }

cufftResult CUFFTAPI cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize) { ava_unsupported; }

cufftResult CUFFTAPI cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t *workSize) { ava_unsupported; }

cufftResult CUFFTAPI cufftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed,
                                       int ostride, int odist, cufftType type, int batch, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftCreate(cufftHandle *handle) { ava_unsupported; }

cufftResult CUFFTAPI cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t *workSize) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist, cufftType type, int batch,
                                      size_t *workArea) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftGetSize(cufftHandle handle, size_t *workSize) { ava_unsupported; }

cufftResult CUFFTAPI cufftSetWorkArea(cufftHandle plan, void *workArea) { ava_unsupported; }

cufftResult CUFFTAPI cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) { ava_unsupported; }

cufftResult CUFFTAPI cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata) { ava_unsupported; }

cufftResult CUFFTAPI cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata) { ava_unsupported; }

cufftResult CUFFTAPI cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata,
                                  int direction) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata) {
  ava_unsupported;
}

cufftResult CUFFTAPI cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata) {
  ava_unsupported;
}

// utility functions
cufftResult CUFFTAPI cufftSetStream(cufftHandle plan, cudaStream_t stream) { ava_unsupported; }

cufftResult CUFFTAPI cufftDestroy(cufftHandle plan) { ava_unsupported; }

cufftResult CUFFTAPI cufftGetVersion(int *version) { ava_unsupported; }

cufftResult CUFFTAPI cufftGetProperty(libraryPropertyType type, int *value) { ava_unsupported; }
#endif // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUFFT_UNIMPLEMENTED_H_
