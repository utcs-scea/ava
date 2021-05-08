#include "common/extensions/tf_optimization.h"

#include <glib.h>
#include <stdint.h>

#include "common/endpoint_lib.h"
#include "common/extensions/cmd_batching.h"

GQueue *call_configuration_stack;
GTree *gpu_address_set;

GQueue *cu_event_pool;
GQueue *idle_cu_event_pool;

cudaError_t cuda_last_error;

// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
#ifdef AVA_PRELOAD_CUBIN
GPtrArray *fatbin_handle_list;
#endif
void worker_tf_opt_init(void) {}

gint gpu_address_range_cmp(gconstpointer r1, gconstpointer r2, gpointer user_data) {
  long diff = ((uintptr_t)r1 - (uintptr_t)r2);
  if (diff < 0) return -1;
  if (diff > 0) return 1;
  return 0;
}

void guestlib_tf_opt_init(void) {
  /* Emulate the call configuration stack */
  call_configuration_stack = g_queue_new();

  /* Save allocated GPU memory addresses */
  gpu_address_set = g_tree_new_full(gpu_address_range_cmp, NULL, NULL, g_free);

  /* Pool descriptors */
  guestlib_cudnn_opt_init();

  cu_event_pool = g_queue_new();
  idle_cu_event_pool = g_queue_new();

  /* API batch */
  nw_global_cmd_batch = cmd_batch_thread_init();
}

void guestlib_tf_opt_fini(void) {
  g_queue_free(call_configuration_stack);
  g_tree_destroy(gpu_address_set);

  /* Free descriptors */
  guestlib_cudnn_opt_fini();

  free_cu_event_pool(cu_event_pool);
  free_cu_event_pool(idle_cu_event_pool);
  g_queue_free(cu_event_pool);
  g_queue_free(idle_cu_event_pool);

  cmd_batch_thread_fini(nw_global_cmd_batch);
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
