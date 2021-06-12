#ifndef _AVA_COMMON_SUPPORT_WORKER_CONTEXT_H_
#define _AVA_COMMON_SUPPORT_WORKER_CONTEXT_H_

#include <glib.h>

#include "common/support/singleton.hpp"
namespace ava {

// global worker context shared by all threads created by worker
class WorkerContext final : public ava::support::Singleton<WorkerContext> {
 public:
  void set_api_server_listen_port(unsigned int p) { api_server_listen_port = p; }

  unsigned int get_api_server_listen_port() const { return api_server_listen_port; }

 private:
  WorkerContext() {
    nw_global_handle_pool = nw_handle_pool_new();
    nw_shadow_thread_pool = shadow_thread_pool_new();

    nw_global_metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
    nw_global_metadata_map = metadata_map_new();
  }

  unsigned int api_server_listen_port;

  /// begin: globals from endpoint lib
  struct nw_handle_pool *nw_global_handle_pool;
  struct shadow_thread_pool_t *nw_shadow_thread_pool;

  pthread_mutex_t nw_global_metadata_map_mutex;
  GHashTable *nw_global_metadata_map;  // guarded by nw_global_metadata_map_mutex
  /// end: globals from endpoint lib
};

}  // namespace ava
#endif  // _AVA_COMMON_SUPPORT_WORKER_CONTEXT_H_
