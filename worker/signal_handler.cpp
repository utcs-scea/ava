#include "signal_handler.h"

#include <absl/debugging/failure_signal_handler.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>

namespace ava_manager {

__sighandler_t original_sigint_handler = SIG_DFL;
__sighandler_t original_sigchld_handler = SIG_DFL;

void sigint_handler(int signo) {
  signal(signo, original_sigint_handler);
  raise(signo);
}

void setupSignalHandlers() {
  if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR) printf("failed to catch SIGINT\n");

  if ((original_sigchld_handler = signal(SIGCHLD, SIG_IGN)) == SIG_ERR) printf("failed to ignore SIGCHLD\n");
  absl::FailureSignalHandlerOptions options;
  options.call_previous_handler = true;
  absl::InstallFailureSignalHandler(options);
}

}  // namespace ava_manager
