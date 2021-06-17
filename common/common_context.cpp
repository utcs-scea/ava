#include "common_context.h"

namespace ava {
CommonContext::CommonContext() {
  nw_global_handle_pool = nw_handle_pool_new();
  nw_shadow_thread_pool = shadow_thread_pool_new();

  nw_global_metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
  nw_global_metadata_map = metadata_map_new();
}
}  // namespace ava
