/**
 * This file implements the handle pooling optimization for TensorFlow 1.14 and
 * ONNXruntime 1.2.0.
 * The underlying dependencies are CUDA 10.1 and cuDNN 7.6.5.
 * The optimization is applied in `cava/samples/onnxruntime/onnx_opt.c`.
 */
#ifndef AVA_EXTENSIONS_TF_OPTIMIZATION_H_
#define AVA_EXTENSIONS_TF_OPTIMIZATION_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <glib.h>

#include "cudnn_optimization.h"

#ifdef __cplusplus
namespace ava {
class GuestContext;
}

void guestlib_tf_opt_init(ava::GuestContext *gctx);
void guestlib_tf_opt_fini(ava::GuestContext *gctx);

extern "C" {
#endif

void worker_tf_opt_init(void);

CUresult __pool_cuEventCreate(CUevent *phEvent, size_t count);
CUresult __pool_cuEventDestroy(CUevent *hEvent, size_t count);
int free_cu_event_pool(GQueue *pool);

CUresult __cuEventQuery(CUevent hEvent);

#ifdef __cplusplus
}
#endif

#endif  // AVA_EXTENSIONS_TF_OPTIMIZATION_H_
