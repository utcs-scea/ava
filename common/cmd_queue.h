#ifndef _AVA_GUESTLIB_QUEUE_WORKER_H_
#define _AVA_GUESTLIB_QUEUE_WORKER_H_

#include "common/cmd_channel.hpp"
#include "common/declaration.h"
#include "common/linkage.h"

namespace ava {

class EXPORTED CmdQueue {
 public:
  CmdQueue();
  virtual ~CmdQueue();
  virtual void enqueue_cmd(::command_base *cmd, ::command_channel *chan, bool is_async) = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(CmdQueue);
};

}  // namespace ava
#endif  // _AVA_GUESTLIB_QUEUE_WORKER_H_
