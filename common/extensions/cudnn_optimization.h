/**
 * This file implements the handle pooling optimization for cuDNN and related APIs
 * used in TensorFlow 1.14 and ONNXruntime 1.2.0.
 * The underlying dependencies are CUDA 10.1 and cuDNN 7.6.5.
 * The optimization is applied in `cava/samples/onnxruntime/onnx_opt.c`.
 */
#ifndef AVA_EXTENSIONS_CUDNN_OPTIMIZATION_H_
#define AVA_EXTENSIONS_CUDNN_OPTIMIZATION_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DESCRITPOR_POOL_SIZE 64
#define CUDNN_HANDLE_POOL_SIZE 2
#define CUBLAS_HANDLE_POOL_SIZE 2

void guestlib_cudnn_opt_init(void);
void guestlib_cudnn_opt_fini(void);
void worker_cudnn_opt_init(void);

cudnnStatus_t __pool_cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count);
int free_convolution_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count);
int free_pooling_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count);
int free_tensor_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count);
int free_filter_descriptor_pool(GQueue *pool);

cudnnStatus_t __cudnnCreate(cudnnHandle_t *handle);
cublasStatus_t __cublasCreate(cublasHandle_t *handle);

#ifdef __cplusplus
}
#endif

#endif  // AVA_EXTENSIONS_CUDNN_OPTIMIZATION_H_
