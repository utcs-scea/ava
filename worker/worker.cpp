#include "worker.h"

#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/singleton.hpp"
#include "common/socket.hpp"
#include "plog/Initializers/RollingFileInitializer.h"
#include "provision_gpu.h"

struct command_channel *chan;
struct command_channel *chan_hv = NULL;
extern int nw_global_vm_id;

__sighandler_t original_sigint_handler = SIG_DFL;
__sighandler_t original_sigsegv_handler = SIG_DFL;
__sighandler_t original_sigchld_handler = SIG_DFL;

void sigint_handler(int signo) {
  void *array[10];
  size_t size;
  size = backtrace(array, 10);
  fprintf(stderr, "===== backtrace =====\n");
  fprintf(stderr, "receive signal %d:\n", signo);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  if (chan) {
    command_channel_free(chan);
    chan = NULL;
  }
  signal(signo, original_sigint_handler);
  raise(signo);
}

void sigsegv_handler(int signo) {
  void *array[10];
  size_t size;
  size = backtrace(array, 10);
  fprintf(stderr, "===== backtrace =====\n");
  fprintf(stderr, "receive signal %d:\n", signo);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  if (chan) {
    command_channel_free(chan);
    chan = NULL;
  }
  signal(signo, original_sigsegv_handler);
  raise(signo);
}

void nw_report_storage_resource_allocation(const char *const name, ssize_t amount) {
  if (chan_hv) command_channel_hv_report_storage_resource_allocation(chan_hv, name, amount);
}

void nw_report_throughput_resource_consumption(const char *const name, ssize_t amount) {
  if (chan_hv) command_channel_hv_report_throughput_resource_consumption(chan_hv, name, amount);
}

static struct command_channel *channel_create() { return chan; }

int main(int argc, char *argv[]) {
  if (!(argc == 3 && !strcmp(argv[1], "migrate")) && (argc != 2)) {
    printf(
        "Usage: %s <listen_port>\n"
        "or     %s <mode> <listen_port> \n",
        argv[0], argv[0]);
    return 0;
  }

  /* Read GPU provision information. */
  char const *cuda_uuid_str = getenv("CUDA_VISIBLE_DEVICES");
  std::string cuda_uuid = cuda_uuid_str ? std::string(cuda_uuid_str) : "";
  char const *gpu_uuid_str = getenv("AVA_GPU_UUID");
  std::string gpu_uuid = gpu_uuid_str ? std::string(gpu_uuid_str) : "";
  char const *gpu_mem_str = getenv("AVA_GPU_MEMORY");
  std::string gpu_mem = gpu_mem_str ? std::string(gpu_mem_str) : "";
  provision_gpu = new ProvisionGpu(cuda_uuid, gpu_uuid, gpu_mem);

  // Initialize logger
  std::string log_file = std::tmpnam(nullptr);
  plog::init(plog::debug, log_file.c_str());

  /* setup signal handler */
  if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR) printf("failed to catch SIGINT\n");

  if ((original_sigsegv_handler = signal(SIGSEGV, sigsegv_handler)) == SIG_ERR) printf("failed to catch SIGSEGV\n");

  if ((original_sigchld_handler = signal(SIGCHLD, SIG_IGN)) == SIG_ERR) printf("failed to ignore SIGCHLD\n");

  /* define arguments */
  auto &setting = ApiServerSetting::instance();
  nw_worker_id = 0;
  unsigned int listen_port;

  /* This is a target API server. Starts live migration */
  if (!strcmp(argv[1], "migrate")) {
    listen_port = (unsigned int)atoi(argv[2]);
    setting.set_listen_port(listen_port);
    std::cerr << "[worker#" << listen_port << "] To check the state of AvA remoting progress, use `tail -f " << log_file
              << "`" << std::endl;

    chan = (struct command_channel *)command_channel_socket_tcp_migration_new(listen_port, 0);
    nw_record_command_channel = command_channel_log_new(listen_port);

    init_internal_command_handler();
    init_command_handler(channel_create);
    LOG_INFO << "[worker#" << listen_port << "] start polling tasks";
    wait_for_command_handler();

    // TODO(migration): connect the guestlib

    return 0;
  }

  /* parse arguments */
  listen_port = (unsigned int)atoi(argv[1]);
  setting.set_listen_port(listen_port);
  std::cerr << "[worker#" << listen_port << "] To check the state of AvA remoting progress, use `tail -f " << log_file
            << "`" << std::endl;

  if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "TCP")) {
    chan_hv = NULL;
    chan = command_channel_socket_tcp_worker_new(listen_port);
    // } else if (!strcmp(getenv("AVA_CHANNEL"), "SHM")) {
    //   chan_hv = command_channel_hv_new(listen_port);
    //   chan = command_channel_shm_worker_new(listen_port);
    // } else if (!strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
    //   chan_hv = command_channel_hv_new(listen_port);
    //   chan = command_channel_socket_worker_new(listen_port);
  } else {
    printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[TCP]\n");
    return 0;
  }

  nw_record_command_channel = command_channel_log_new(listen_port);
  init_internal_command_handler();
  init_command_handler(channel_create);
  LOG_INFO << "[worker#" << listen_port << "] start polling tasks";
  wait_for_command_handler();
  command_channel_free(chan);
  command_channel_free((struct command_channel *)nw_record_command_channel);
  if (chan_hv) command_channel_hv_free(chan_hv);

  return 0;
}
