#ifndef _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDNN_UNIMPLEMENTED_H_
#define _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDNN_UNIMPLEMENTED_H_
#include <cudnn.h>

#include "cava/nightwatch/parser/c/nightwatch.h"

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                const int hiddenSize, const int numLayers,
                                                cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
                                                cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize,
                                                int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
                                                cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction,
                                                cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                          cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt, double lclip,
                                          double rclip) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                          cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt,
                                          double *lclip, double *rclip) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                      const int recProjSize, const int outProjSize) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                      int *recProjSize, int *outProjSize) {
  ava_unsupported;
}

/* Expensive. Creates the plan for the specific settings. */
cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, const int minibatch,
                                                       const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan) {
  ava_unsupported;
}

/* dataType in weight descriptors and input descriptors is used to describe storage */
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength, const cudnnTensorDescriptor_t *xDesc,
                                                   size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                         const int seqLength, const cudnnTensorDescriptor_t *xDesc,
                                                         size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
                                                cudnnDataType_t dataType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                          const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
                                                          const cudnnFilterDescriptor_t wDesc, const void *w,
                                                          const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
                                                          void **linLayerMat) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                        const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
                                                        const cudnnFilterDescriptor_t wDesc, const void *w,
                                                        const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
                                                        void **linLayerBias) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                  const int seqLength, const cudnnTensorDescriptor_t *xDesc,
                                                  const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx,
                                                  const cudnnTensorDescriptor_t *yDesc, const void *y,
                                                  const void *workspace, size_t workSpaceSizeInBytes,
                                                  const cudnnFilterDescriptor_t dwDesc, void *dw,
                                                  const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

/* RNN EX API */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType, cudnnRNNDataLayout_t layout, int maxSeqLength,
    int batchSize, int vectorSize, const int seqLengthArray[], /* length of each sequence in the batch */
    void *paddingFill)                                         /* symbol for filling padding position in output */
{
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
                                                    cudnnRNNDataLayout_t *layout, int *maxSeqLength, int *batchSize,
                                                    int *vectorSize, int arrayLengthRequested, int seqLengthArray[],
                                                    void *paddingFill) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
    const void *keys,                     /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
    void *cAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
    void *iAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
    void *queries,                        /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInferenceEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
    const void *keys,                     /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
    void *cAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
    void *iAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
    void *queries,                        /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardDataEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, /* reserved, should pass NULL */
    const void *dcAttn,                    /* reserved, should pass NULL */
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, /* reserved, should pass NULL */
    void *dkeys,                           /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeightsEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                    const cudnnRNNDataDescriptor_t xDesc, const void *x,
                                                    const cudnnTensorDescriptor_t hxDesc, const void *hx,
                                                    const cudnnRNNDataDescriptor_t yDesc, const void *y,
                                                    void *workSpace, size_t workSpaceSizeInBytes,
                                                    const cudnnFilterDescriptor_t dwDesc, void *dw, void *reserveSpace,
                                                    size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

/* RNN FIND API */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                         cudnnAlgorithmDescriptor_t algoDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle,
                                                                       const cudnnRNNDescriptor_t rnnDesc, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle,
                                                                      const cudnnRNNDescriptor_t rnnDesc, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                                   const cudnnRNNDescriptor_t rnnDesc, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle,
                                                                      const cudnnRNNDescriptor_t rnnDesc, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

/* DEPRECATED routines to be removed next release :
   User should use the non-suffixed version (which has the API and functionality of _v6 version)
   Routines with _v5 suffix has the functionality of the non-suffixed routines in the CUDNN V6
 */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                   const int hiddenSize, const int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                   cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                   cudnnDataType_t mathPrec) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
                                                     cudnnDataType_t dataType, /* image data type */
                                                     int n,                    /* number of inputs (batch size) */
                                                     int c,                    /* number of input feature maps */
                                                     int h,                    /* height of input section */
                                                     int w)                    /* width of input section */
{
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                       cudnnDataType_t dataType, /* image data type */
                                                       int n,                    /* number of inputs (batch size) */
                                                       int c,                    /* number of input feature maps */
                                                       int h,                    /* height of input section */
                                                       int w,                    /* width of input section */
                                                       int nStride, int cStride, int hStride, int wStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                                                     cudnnDataType_t *dataType, /* image data type */
                                                     int *n,                    /* number of inputs (batch size) */
                                                     int *c,                    /* number of input feature maps  */
                                                     int *h,                    /* height of input section */
                                                     int *w,                    /* width of input section */
                                                     int *nStride, int *cStride, int *hStride, int *wStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
                                                     cudnnDataType_t *dataType, int *nbDims, int dimA[],
                                                     int strideA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
  ava_unsupported;
}

/** Create an empty tensor transform descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc) {
  ava_unsupported;
}

/** Initialize a previously created tensor transform descriptor. */
cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                            const uint32_t nbDims, const cudnnTensorFormat_t destFormat,
                                                            const int32_t padBeforeA[], const int32_t padAfterA[],
                                                            const uint32_t foldA[],
                                                            const cudnnFoldingDirection_t direction) {
  ava_unsupported;
}

/**
 * Retrieves the values stored in a previously initialized tensor transform
 * descriptor.
 */
cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                            uint32_t nbDimsRequested, cudnnTensorFormat_t *destFormat,
                                                            int32_t padBeforeA[], int32_t padAfterA[], uint32_t foldA[],
                                                            cudnnFoldingDirection_t *direction) {
  ava_unsupported;
}

/**
 * Destroys a previously created tensor transform descriptor.
 */
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc) {
  ava_unsupported;
}

/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle, const void *alpha,
                                               const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                               const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
                                                 const void *alpha, const cudnnTensorDescriptor_t srcDesc,
                                                 const void *srcData, const void *beta,
                                                 const cudnnTensorDescriptor_t destDesc, void *destData) {
  ava_unsupported;
}

/* Helper function to calculate folding descriptors  for dgrad */
cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat, cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc, cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc, cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc, cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
  ava_unsupported;
}

/** Create a destination descriptor for cudnnTransformTensor */
cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                                                 const cudnnTensorDescriptor_t srcDesc,
                                                 cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                                     cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType,
                                                     cudnnNanPropagation_t opTensorNanOpt) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                                                     cudnnOpTensorOp_t *opTensorOp, cudnnDataType_t *opTensorCompType,
                                                     cudnnNanPropagation_t *opTensorNanOpt) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) { ava_unsupported; }

/* Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/* B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
cudnnStatus_t CUDNNWINAPI cudnnOpTensor(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
                                        const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
                                        const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
                                        const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         cudnnReduceTensorOp_t reduceTensorOp,
                                                         cudnnDataType_t reduceTensorCompType,
                                                         cudnnNanPropagation_t reduceTensorNanOpt,
                                                         cudnnReduceTensorIndices_t reduceTensorIndices,
                                                         cudnnIndicesType_t reduceTensorIndicesType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         cudnnReduceTensorOp_t *reduceTensorOp,
                                                         cudnnDataType_t *reduceTensorCompType,
                                                         cudnnNanPropagation_t *reduceTensorNanOpt,
                                                         cudnnReduceTensorIndices_t *reduceTensorIndices,
                                                         cudnnIndicesType_t *reduceTensorIndicesType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  ava_unsupported;
}

/* Helper function to return the minimum size of the index space to be passed to the reduction given the input and
 * output tensors */
cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                                       const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                       const cudnnTensorDescriptor_t aDesc,
                                                       const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

/* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output
 * tensors */
cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                                         const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         const cudnnTensorDescriptor_t aDesc,
                                                         const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

/* Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                            void *indices, size_t indicesSizeInBytes, void *workspace,
                                            size_t workspaceSizeInBytes, const void *alpha,
                                            const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
                                            const cudnnTensorDescriptor_t cDesc, void *C) {
  ava_unsupported;
}

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                         const void *valuePtr) {
  ava_unsupported;
}

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                           const void *alpha) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                     cudnnDataType_t dataType, /* image data type */
                                                     cudnnTensorFormat_t format,
                                                     int k, /* number of output feature maps */
                                                     int c, /* number of input feature maps */
                                                     int h, /* height of each input filter */
                                                     int w) {
  ava_unsupported;
} /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                                                     cudnnDataType_t *dataType, /* image data type */
                                                     cudnnTensorFormat_t *format,
                                                     int *k, /* number of output feature maps */
                                                     int *c, /* number of input feature maps */
                                                     int *h, /* height of each input filter */
                                                     int *w) {
  ava_unsupported;
} /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
                                                     cudnnDataType_t *dataType, /* image data type */
                                                     cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
  ava_unsupported;
}
cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnTransformFilter(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
                                               const void *alpha, const cudnnFilterDescriptor_t srcDesc,
                                               const void *srcData, const void *beta,
                                               const cudnnFilterDescriptor_t destDesc, void *destData) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
                                                    cudnnReorderType_t reorderType, const void *filterData,
                                                    void *reorderedFilterData, int reorderBias, const void *biasData,
                                                    void *reorderedBiasData) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnMathType_t *mathType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                         cudnnReorderType_t reorderType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                         cudnnReorderType_t *reorderType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h, /* zero-padding height */
                                int pad_w,                                        /* zero-padding width */
                                int u,                                            /* vertical filter stride */
                                int v,                                            /* horizontal filter stride */
                                int dilation_h, /* filter dilation in the vertical dimension */
                                int dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc, int *pad_h, /* zero-padding height */
                                int *pad_w,                                              /* zero-padding width */
                                int *u,                                                  /* vertical filter stride */
                                int *v,                                                  /* horizontal filter stride */
                                int *dilation_h, /* filter dilation in the vertical dimension */
                                int *dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  ava_unsupported;
}

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                                const cudnnFilterDescriptor_t filterDesc, int *n,
                                                                int *c, int *h, int *w) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                                          int arrayLengthRequested, int *arrayLength, int padA[],
                                                          int strideA[], int dilationA[], cudnnConvolutionMode_t *mode,
                                                          cudnnDataType_t *computeType) {
  ava_unsupported;
} /* convolution data type */

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                                const cudnnFilterDescriptor_t filterDesc, int nbDims,
                                                                int tensorOuputDimA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  ava_unsupported;
}

/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void *alpha,
                                                       const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                       const void *beta, const cudnnTensorDescriptor_t dbDesc,
                                                       void *db) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t *algo) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
                                      const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
                                      void *colBuffer) {
  ava_unsupported;
}

/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                              cudnnSoftmaxMode_t mode, const void *alpha,
                                              const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                              const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                               cudnnSoftmaxMode_t mode, const void *alpha,
                                               const cudnnTensorDescriptor_t yDesc, const void *y,
                                               const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
                                               const cudnnTensorDescriptor_t dxDesc, void *dx) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
                                                      cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
                                                      int windowWidth, int verticalPadding, int horizontalPadding,
                                                      int verticalStride, int horizontalStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                                      cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
                                                      int *windowHeight, int *windowWidth, int *verticalPadding,
                                                      int *horizontalPadding, int *verticalStride,
                                                      int *horizontalStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
                                                      cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
                                                      int *nbDims, int windowDimA[], int paddingA[], int strideA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                                            const cudnnTensorDescriptor_t inputTensorDesc, int nbDims,
                                                            int outputTensorDimA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                                            const cudnnTensorDescriptor_t inputTensorDesc, int *n,
                                                            int *c, int *h, int *w) {
  ava_unsupported;
}

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                                       cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt,
                                                       double coef) {
  ava_unsupported;
} /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                                       cudnnActivationMode_t *mode, cudnnNanPropagation_t *reluNanOpt,
                                                       double *coef) {
  ava_unsupported;
} /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  ava_unsupported;
}

/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
                                                 const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                                 const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
                                                  const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
                                                  const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                  const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                                  const cudnnTensorDescriptor_t dxDesc, void *dx) {
  ava_unsupported;
}

/*
 * Create an instance of LRN (Local Response Normalization) descriptor
 * Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
 */
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) { ava_unsupported; }

/*
 * Uses a window [center-lookBehind, center+lookAhead], where
 * lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
 * Values of double parameters cast to tensor data type.
 */
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha,
                                                double lrnBeta, double lrnK) {
  ava_unsupported;
}
/*
 * Retrieve the settings currently stored in an LRN layer descriptor
 * Any of the provided pointers can be NULL (no corresponding value will be returned)
 */
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha,
                                                double *lrnBeta, double *lrnK) {
  ava_unsupported;
}

/* Destroy an instance of LRN descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) { ava_unsupported; }

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
                                                      cudnnLRNMode_t lrnMode, const void *alpha,
                                                      const cudnnTensorDescriptor_t xDesc, const void *x,
                                                      const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
                                                       cudnnLRNMode_t lrnMode, const void *alpha,
                                                       const cudnnTensorDescriptor_t yDesc, const void *y,
                                                       const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                       const cudnnTensorDescriptor_t xDesc, const void *x,
                                                       const void *beta, const cudnnTensorDescriptor_t dxDesc,
                                                       void *dx) {
  ava_unsupported;
}

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
    const void *x, const void *means,    /* if NULL, means are assumed to be zero */
    void *temp, void *temp2, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, means, dy, temp, temp2 */
    const void *x, const void *means,    /* if NULL, means are assumed to be zero */
    const void *dy, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
    void *dx,                                   /* output x differential */
    void *dMeans) {
  ava_unsupported;
} /* output means differential, can be NULL */

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                                                 cudnnSamplerType_t samplerType,
                                                                 cudnnDataType_t dataType, const int nbDims,
                                                                 const int dimA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                                             const cudnnSpatialTransformerDescriptor_t stDesc,
                                                             const void *theta, void *grid) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                                              const cudnnSpatialTransformerDescriptor_t stDesc,
                                                              const void *dgrid, void *dtheta) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
                                                       const void *alpha, const cudnnTensorDescriptor_t xDesc,
                                                       const void *x, const void *grid, const void *beta,
                                                       cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
                                                        cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha,
                                                        const cudnnTensorDescriptor_t xDesc, const void *x,
                                                        const void *beta, const cudnnTensorDescriptor_t dxDesc,
                                                        void *dx, const void *alphaDgrid,
                                                        const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                        const void *grid, const void *betaDgrid, void *dgrid) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) { ava_unsupported; }

/*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes) { ava_unsupported; }

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                    float dropout, void *states, size_t stateSizeInBytes,
                                                    unsigned long long seed) {
  ava_unsupported;
}

/* Restores the dropout descriptor to a previously saved-off state */
cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                        float dropout, void *states, size_t stateSizeInBytes,
                                                        unsigned long long seed) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                    float *dropout, void **states, unsigned long long *seed) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
                                              const cudnnTensorDescriptor_t xdesc, const void *x,
                                              const cudnnTensorDescriptor_t ydesc, void *y, void *reserveSpace,
                                              size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
                                               const cudnnTensorDescriptor_t dydesc, const void *dy,
                                               const cudnnTensorDescriptor_t dxdesc, void *dx, void *reserveSpace,
                                               size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode) {
  ava_unsupported;
}

/* Sequence data descriptor */

cudnnStatus_t CUDNNWINAPI cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
                                                    int nbDims, const int dimA[], const cudnnSeqDataAxis_t axes[],
                                                    size_t seqLengthArraySize, const int seqLengthArray[],
                                                    void *paddingFill) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                                                    cudnnDataType_t *dataType, int *nbDims, int nbDimsRequested,
                                                    int dimA[], cudnnSeqDataAxis_t axes[], size_t *seqLengthArraySize,
                                                    size_t seqLengthSizeRequested, int seqLengthArray[],
                                                    void *paddingFill) {
  ava_unsupported;
}

/* Multihead Attention */

/* Multi-head attention modes set in attention descriptor */

cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads,
                                                 double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec,
                                                 cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
                                                 cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize,
                                                 int vSize, int qProjSize, int kProjSize, int vProjSize, int oProjSize,
                                                 int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize,
                                                 int maxBeamSize) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads, double *smScaler, cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec, cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize, int *vSize, int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize, int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize, int *maxBeamSize) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
                                                       size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
                                                       size_t *reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
                                                       cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
                                                       const void *weights, cudnnTensorDescriptor_t wDesc,
                                                       void **wAddr) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx, const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsQO[], const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc, const void *queries,
    const void *residuals, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries, const void *queries,
    const cudnnSeqDataDescriptor_t dkDesc, void *dkeys, const void *keys, const cudnnSeqDataDescriptor_t dvDesc,
    void *dvalues, const void *values, size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes,
    void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc, const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *weights, void *dweights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  ava_unsupported;
}

/*
 * CTC (Connectionist Temporal Classification) loss descriptor create/destory/set/get functions
 */
cudnnStatus_t CUDNNWINAPI cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
                                                      cudnnLossNormalizationMode_t normMode,
                                                      cudnnNanPropagation_t gradMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
                                                      cudnnLossNormalizationMode_t *normMode,
                                                      cudnnNanPropagation_t *gradMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) { ava_unsupported; }

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size)  */
    const void *probs,                       /* probabilities after softmax, in GPU memory */
    const int *labels,                       /* labels, in CPU memory */
    const int *labelLengths,                 /* the length of each label, in CPU memory */
    const int *inputLengths,                 /* the lengths of timing steps in each batch, in CPU memory */
    void *costs,                             /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
    const void *gradients,   /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes) {
  ava_unsupported;
} /* size of the workspace */

/* return the workspace size needed for ctc */
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size) */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the
                                                    dimensions are T,N,A. To compute costs
                                                    only, set it to NULL */
    const int *labels,                           /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    cudnnCTCLossAlgo_t algo,                     /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
  ava_unsupported;
} /* pointer to the returned workspace size */

cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc,
                                                      cudnnAlgorithm_t *algorithm) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src,
                                                       cudnnAlgorithmDescriptor_t dest) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                                                       cudnnAlgorithmDescriptor_t algoDesc, cudnnStatus_t status,
                                                       float time, size_t memory) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                                                       cudnnAlgorithmDescriptor_t *algoDesc, cudnnStatus_t *status,
                                                       float *time, size_t *memory) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
                                                     size_t *algoSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSaveAlgorithm(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, void *algoSpace,
                                             size_t algoSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(cudnnHandle_t handle, void *algoSpace, size_t algoSpaceSizeInBytes,
                                                cudnnAlgorithmDescriptor_t algoDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack,
                                                            cudnnFusedOps_t ops) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,
                                                                  cudnnFusedOpsConstParamLabel_t paramLabel,
                                                                  const void *param) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                                                  cudnnFusedOpsConstParamLabel_t paramLabel,
                                                                  void *param, int *isNULL) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack,
                                                              cudnnFusedOps_t ops) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                                                    cudnnFusedOpsVariantParamLabel_t paramLabel,
                                                                    void *ptr) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                                                    cudnnFusedOpsVariantParamLabel_t paramLabel,
                                                                    void *ptr) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                                                const cudnnFusedOpsConstParamPack_t constPack,
                                                size_t *workspaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
                                               cudnnFusedOpsVariantParamPack_t varPack) {
  ava_unsupported;
}

#endif  // _AVA_CAVA_SAMPLES_CUDA_COMMON_SPEC_CUDNN_UNIMPLEMENTED_H_
