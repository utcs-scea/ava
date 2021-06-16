#ifndef _AVA_COMMON_COMMON_CONTEXT_H_
#define _AVA_COMMON_COMMON_CONTEXT_H_

#include <glib.h>

#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/shadow_thread_pool.hpp"
#include "common/support/singleton.hpp"

namespace ava {
class EXPORTED CommonContext final : public ava::support::Singleton<CommonContext> {
 public:
  friend class ava::support::Singleton<CommonContext>;
  // globals from endpoint_lib.cpp
  struct nw_handle_pool *nw_global_handle_pool;
  struct shadow_thread_pool_t *nw_shadow_thread_pool;

  pthread_mutex_t nw_global_metadata_map_mutex;
  GHashTable *nw_global_metadata_map;  // guarded by nw_global_metadata_map_mutex

 private:
  CommonContext();
};

}  // namespace ava

#endif  // _AVA_COMMON_COMMON_CONTEXT_H_
