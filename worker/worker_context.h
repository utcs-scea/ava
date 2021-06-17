#ifndef _AVA_COMMON_SUPPORT_WORKER_CONTEXT_H_
#define _AVA_COMMON_SUPPORT_WORKER_CONTEXT_H_

#ifndef AVA_WORKER
#error "This file should only be included by worker"
#else
#include <glib.h>

#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/shadow_thread_pool.hpp"
#include "common/support/singleton.hpp"

namespace ava {

// global worker context shared by all threads created by worker
class EXPORTED WorkerContext final : public ava::support::Singleton<WorkerContext> {
 public:
  friend class ava::support::Singleton<WorkerContext>;
  void set_api_server_listen_port(unsigned int p) { api_server_listen_port = p; }

  unsigned int get_api_server_listen_port() const { return api_server_listen_port; }

 private:
  WorkerContext();

  unsigned int api_server_listen_port;
};

}  // namespace ava
#endif  // AVA_WORKER

#endif  // _AVA_COMMON_SUPPORT_WORKER_CONTEXT_H_
