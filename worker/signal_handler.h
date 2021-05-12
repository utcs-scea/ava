#ifndef AVA_WORKER_SIGNAL_HANDLER_H_
#define AVA_WORKER_SIGNAL_HANDLER_H_

#include <signal.h>

namespace ava_manager {

void sigint_handler(int signo);
void setupSignalHandlers();

}  // namespace ava_manager

#endif  // AVA_WORKER_SIGNAL_HANDLER_H_
