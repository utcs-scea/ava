#ifndef _AVA_GUESTLIB_COMMAND_WORKER_H_
#define _AVA_GUESTLIB_COMMAND_WORKER_H_

#include <glib.h>

#include "common/cmd_channel.hpp"
#include "common/cmd_queue.h"
#include "common/support/thread.h"
#include "guestlib/guest_thread.h"

namespace ava {

class GuestCmdBatchingQueue : public CmdQueue {
 public:
  GuestCmdBatchingQueue();
  ~GuestCmdBatchingQueue() override;

  void Start();
  void Stop();
  void enqueue_cmd(::command_base *cmd, ::command_channel *chan, bool is_async) override;

 private:
  bool running_;
  GuestThread cmd_batching_thread_;
  GAsyncQueue *pending_cmds_;
  GAsyncQueue *active_cmds_;
  void CmdBatchingThreadMain();
  DISALLOW_COPY_AND_ASSIGN(GuestCmdBatchingQueue);
};

}  // namespace ava

#endif  // _AVA_GUESTLIB_COMMAND_WORKER_H_
