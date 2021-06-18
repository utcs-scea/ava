#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDNN_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDNN_H_
#include <cudnn.h>

#include "cava/nightwatch/parser/c/nightwatch.h"

/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t aDesc,
                                         const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc,
                                         void *C) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(aDesc) ava_handle;
  ava_argument(A) ava_opaque;
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(cDesc) ava_handle;
  ava_argument(C) ava_opaque;
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(wDesc) ava_handle;
  ava_argument(w) ava_opaque;
  ava_argument(convDesc) ava_handle;
  ava_argument(yDesc) ava_handle;
  ava_argument(y) ava_opaque;
  ava_argument(returnedAlgoCount) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(perfResults) {
    ava_out;
    cu_in_out_buffer(requestedAlgoCount, returnedAlgoCount);
  }
  ava_argument(workSpace) ava_opaque;
}

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(dyDesc) ava_handle;
  ava_argument(convDesc) ava_handle;
  ava_argument(gradDesc) ava_handle;
  ava_argument(sizeInBytes) {
    ava_out;
    ava_buffer(1);
  }
}

/*
 * Derives a tensor descriptor from layer data descriptor for BatchNormalization
 * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
 * bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                                        const cudnnTensorDescriptor_t xDesc,
                                                        cudnnBatchNormMode_t mode) {
  ava_async;
  ava_argument(derivedBnDesc) ava_handle;
  ava_argument(xDesc) ava_handle;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDNN_H_
