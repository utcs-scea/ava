#include "common/extensions/tf_optimization.h"

#include <glib.h>
#include <stdint.h>

#include "common/endpoint_lib.hpp"
#include "guestlib/extensions/gpu_address_tracking.h"
#include "guestlib/extensions/guest_cmd_batching_queue.h"
#include "guestlib/guest_context.h"
#include "guestlib/guest_thread.h"

GQueue *call_configuration_stack;

GQueue *cu_event_pool;
GQueue *idle_cu_event_pool;

cudaError_t cuda_last_error;

// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
#ifdef AVA_PRELOAD_CUBIN
GPtrArray *fatbin_handle_list;
#endif
void worker_tf_opt_init(void) {}

void guestlib_tf_opt_init(ava::GuestContext *gctx) {
  /* Emulate the call configuration stack */
  call_configuration_stack = g_queue_new();

  gpu_address_tracking_init();
  /* Pool descriptors */
  guestlib_cudnn_opt_init();

  cu_event_pool = g_queue_new();
  idle_cu_event_pool = g_queue_new();

  /* API batch */
  gctx->guest_cmd_batching_queue_ = new ava::GuestCmdBatchingQueue();
  gctx->guest_cmd_batching_queue_->Start();
}

void guestlib_tf_opt_fini(ava::GuestContext *gctx) {
  g_queue_free(call_configuration_stack);
  gpu_address_tracking_fini();

  /* Free descriptors */
  guestlib_cudnn_opt_fini();

  free_cu_event_pool(cu_event_pool);
  free_cu_event_pool(idle_cu_event_pool);
  g_queue_free(cu_event_pool);
  g_queue_free(idle_cu_event_pool);

  gctx->guest_cmd_batching_queue_->Stop();
  delete gctx->guest_cmd_batching_queue_;
}

int free_cu_event_pool(GQueue *pool) {
  gpointer element;
  CUevent *desc;
  int i = 0;

  if (g_queue_is_empty(pool)) return CUDA_SUCCESS;

  desc = (CUevent *)malloc(sizeof(CUevent) * pool->length);

  while ((element = g_queue_pop_head(pool))) {
    desc[i++] = (CUevent)element;
  }

  return __pool_cuEventDestroy(desc, i);
}
