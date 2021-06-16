#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_CUBLAS1_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_BLAS1_UNIMPLEMENTED_H_
#include <cublas_api.h>
#include <cublas_v2.h>

/* ---------------- CUBLAS BLAS1 functions ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                   int incx, void *result, cudaDataType resultType,
                                                   cudaDataType executionType) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                     float *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                     double *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                      float *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                      double *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                  int incx, const void *y, cudaDataType yType, int incy, void *result,
                                                  cudaDataType resultType, cudaDataType executionType) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                   int incx, const void *y, cudaDataType yType, int incy, void *result,
                                                   cudaDataType resultType, cudaDataType executionType) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                    const float *y, int incy,
                                                    float *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                    const double *y, int incy,
                                                    double *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                     const cuComplex *y, int incy,
                                                     cuComplex *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                     const cuComplex *y, int incy,
                                                     cuComplex *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                     const cuDoubleComplex *y, int incy,
                                                     cuDoubleComplex *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                     const cuDoubleComplex *y, int incy,
                                                     cuDoubleComplex *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle, int n,
                                                   const void *alpha, /* host or device pointer */
                                                   cudaDataType alphaType, void *x, cudaDataType xType, int incx,
                                                   cudaDataType executionType) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     double *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, int n,
                                                      const float *alpha, /* host or device pointer */
                                                      cuComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, int n,
                                                      const double *alpha, /* host or device pointer */
                                                      cuDoubleComplex *x, int incx) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx(cublasHandle_t handle, int n,
                                                   const void *alpha, /* host or device pointer */
                                                   cudaDataType alphaType, const void *x, cudaDataType xType, int incx,
                                                   void *y, cudaDataType yType, int incy, cudaDataType executiontype) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle, int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *x, int incx, float *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle, int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *x, int incx, double *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle, int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x, int incx, cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle, int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCopyEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                   int incx, void *y, cudaDataType yType, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y,
                                                     int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y,
                                                     int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                     cuComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx, float *y,
                                                     int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx, double *y,
                                                     int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y,
                                                     int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                                                     cuDoubleComplex *y, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSwapEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx,
                                                   void *y, cudaDataType yType, int incy) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                      int *result) /* host or device pointer */

{
  ava_unsupported;
}
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIamaxEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                    int incx, int *result /* host or device pointer */
) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                      int *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIaminEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                    int incx, int *result /* host or device pointer */
) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAsumEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                   int incx, void *result,
                                                   cudaDataType resultType, /* host or device pointer */
                                                   cudaDataType executiontype) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                     float *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                     double *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                      float *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
                                                      double *result) /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx, float *y,
                                                    int incy, const float *c, /* host or device pointer */
                                                    const float *s)           /* host or device pointer */
{
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx, double *y,
                                                    int incy, const double *c, /* host or device pointer */
                                                    const double *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y,
                                                    int incy, const float *c, /* host or device pointer */
                                                    const cuComplex *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y,
                                                     int incy, const float *c, /* host or device pointer */
                                                     const float *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                                                    cuDoubleComplex *y, int incy,
                                                    const double *c, /* host or device pointer */
                                                    const cuDoubleComplex *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                                                     cuDoubleComplex *y, int incy,
                                                     const double *c, /* host or device pointer */
                                                     const double *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx,
                                                  void *y, cudaDataType yType, int incy,
                                                  const void *c, /* host or device pointer */
                                                  const void *s, cudaDataType csType, cudaDataType executiontype) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, float *a, /* host or device pointer */
                                                     float *b,                        /* host or device pointer */
                                                     float *c,                        /* host or device pointer */
                                                     float *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, double *a, /* host or device pointer */
                                                     double *b,                        /* host or device pointer */
                                                     double *c,                        /* host or device pointer */
                                                     double *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, /* host or device pointer */
                                                     cuComplex *b,                        /* host or device pointer */
                                                     float *c,                            /* host or device pointer */
                                                     cuComplex *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle,
                                                     cuDoubleComplex *a, /* host or device pointer */
                                                     cuDoubleComplex *b, /* host or device pointer */
                                                     double *c,          /* host or device pointer */
                                                     cuDoubleComplex *s) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotgEx(cublasHandle_t handle, void *a, /* host or device pointer */
                                                   void *b,                        /* host or device pointer */
                                                   cudaDataType abType, void *c,   /* host or device pointer */
                                                   void *s,                        /* host or device pointer */
                                                   cudaDataType csType, cudaDataType executiontype) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx, float *y,
                                                     int incy, const float *param) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx, double *y,
                                                     int incy, const double *param) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx,
                                                   void *y, cudaDataType yType, int incy,
                                                   const void *param, /* host or device pointer */
                                                   cudaDataType paramType, cudaDataType executiontype) {
  ava_unsupported;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, float *d1, /* host or device pointer */
                                                      float *d2,                        /* host or device pointer */
                                                      float *x1,                        /* host or device pointer */
                                                      const float *y1,                  /* host or device pointer */
                                                      float *param) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, double *d1, /* host or device pointer */
                                                      double *d2,                        /* host or device pointer */
                                                      double *x1,                        /* host or device pointer */
                                                      const double *y1,                  /* host or device pointer */
                                                      double *param) {
  ava_unsupported;
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmgEx(cublasHandle_t handle, void *d1,     /* host or device pointer */
                                                    cudaDataType d1Type, void *d2,       /* host or device pointer */
                                                    cudaDataType d2Type, void *x1,       /* host or device pointer */
                                                    cudaDataType x1Type, const void *y1, /* host or device pointer */
                                                    cudaDataType y1Type, void *param,    /* host or device pointer */
                                                    cudaDataType paramType, cudaDataType executiontype) {
  ava_unsupported;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUBLAS_CUBLAS1_UNIMPLEMENTED_H_
