#ifndef __EXECUTOR_WORKER_H__
#define __EXECUTOR_WORKER_H__

#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

#include "common/cmd_channel.hpp"

#ifdef __cplusplus
#include <plog/Log.h>

#include <vector>

extern "C" {
#endif

typedef struct MemoryRegion {
  void *addr;
  size_t size;
} MemoryRegion;

void nw_report_storage_resource_allocation(const char *const name, ssize_t amount);
void nw_report_throughput_resource_consumption(const char *const name, ssize_t amount);

/* For Python wrapper */
int init_manager_vsock();
struct command_base *poll_client(int listen_fd, int *client_fd, int *guest_cid);
void respond_client(int client_fd, int worker_id);
void close_client(int client_fd);

#ifdef __cplusplus
}

struct command_channel *command_channel_socket_tcp_worker_new(int worker_port);
struct command_channel *command_channel_shm_worker_new(int listen_port);
#endif

#endif
