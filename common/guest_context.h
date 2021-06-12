#ifndef _AVA_COMMON_GUEST_CONTEXT_H_
#define _AVA_COMMON_GUEST_CONTEXT_H_

#include <glib.h>

#include "common/support/singleton.hpp"

namespace ava {

// global guest context shared by all threads created by guestlib
class GuestContext final : public Singleton<GuestContext> {
 public:
 private:
  GuestContext() {
    nw_global_handle_pool = nw_handle_pool_new();
    nw_shadow_thread_pool = shadow_thread_pool_new();

    nw_global_metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
    nw_global_metadata_map = metadata_map_new();
  }

  // globals from endpoint_lib.cpp
  struct nw_handle_pool *nw_global_handle_pool;
  struct shadow_thread_pool_t *nw_shadow_thread_pool;

  pthread_mutex_t nw_global_metadata_map_mutex;
  GHashTable *nw_global_metadata_map;  // guarded by nw_global_metadata_map_mutex

  // global from guestlib/extensions/cmd_batching.cpp
  struct command_batch *nw_global_cmd_batch{nullptr};

  // global from guestlib/extensions/cudnn_optimization.cpp
  GQueue *convolution_descriptor_pool;
  GQueue *idle_convolution_descriptor_pool;
  GQueue *pooling_descriptor_pool;
  GQueue *idle_pooling_descriptor_pool;
  GQueue *tensor_descriptor_pool;
  GQueue *idle_tensor_descriptor_pool;
  GQueue *filter_descriptor_pool;
  GQueue *idle_filter_descriptor_pool;

  // global from guestlib/extensions/tf_optimization.cpp
  GQueue *call_configuration_stack;
  GTree *gpu_address_set;

  GQueue *cu_event_pool;
  GQueue *idle_cu_event_pool;
};

}  // namespace ava

#endif  // _AVA_COMMON_GUEST_CONTEXT_H_
