#ifndef _AVA_COMMON_GUEST_CONTEXT_H_
#define _AVA_COMMON_GUEST_CONTEXT_H_

#ifndef AVA_GUESTLIB
#error "This file should only be included by guestlib"
#else

#include <absl/container/flat_hash_map.h>
#include <glib.h>

#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/shadow_thread_pool.hpp"
#include "common/support/singleton.hpp"
#include "guestlib/extensions/guest_cmd_batching_queue.h"

namespace ava {

// global guest context shared by all threads created by guestlib
class EXPORTED GuestContext final : public ava::support::Singleton<GuestContext> {
 public:
  friend class ava::support::Singleton<GuestContext>;

  ~GuestContext();
  ava::GuestCmdBatchingQueue *guest_cmd_batching_queue_{nullptr};

 private:
  GuestContext();
};

}  // namespace ava

#endif  // AVA_GUESTLIB

#endif  // _AVA_COMMON_GUEST_CONTEXT_H_
