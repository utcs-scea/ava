#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_channel_socket_utilities.hpp"
#include "common/cmd_handler.hpp"
#include "common/logging.h"
#include "common/support/io.h"
#include "common/support/socket.h"
#include "guest_config.h"
#include "guestlib.h"
#include "manager_service.proto.h"
#include <absl/strings/str_split.h>
#include <fmt/format.h>

namespace {
extern struct command_channel_vtable command_channel_socket_tcp_vtable;
}

extern int nw_global_vm_id;

/**
 * TCP channel guestlib endpoint.
 *
 * The `manager_tcp` is required to use the TCP channel.
 */
std::vector<struct command_channel *> command_channel_socket_tcp_guest_new() {
  // Connect API server manager
  std::vector<std::string> manager_addr = absl::StrSplit(guestconfig::config->manager_address_, absl::ByAnyChar(":-/ "));
  DCHECK(manager_addr.size() == 2) << "Invalid API server manager address";
  struct sockaddr_in addr;
  if (!ava::support::ResolveTcpAddr(&addr, manager_addr[0], manager_addr[1])) {
    AVA_LOG(FATAL) << fmt::format("Cannot resolve manager address {}:{}", manager_addr[0], manager_addr[1]);
    abort();
  }

  int manager_sock = ava::support::TcpSocketConnect(&addr);
  if (manager_sock == -1) {
    AVA_LOG(FATAL) << "Cannot connect to manager";
    abort();
  }

  // Serialize configurations
  ava_proto::WorkerAssignRequest request;
  request.gpu_count() = guestconfig::config->gpu_memory_.size();
  for (auto m : guestconfig::config->gpu_memory_) {
    request.gpu_mem().push_back(m << 20);
  }
  std::vector<unsigned char> request_buf;
  zpp::serializer::memory_output_archive out(request_buf);
  out(request);
  uint32_t request_length = static_cast<uint32_t>(request_buf.size());
  if (!ava::support::SendData(manager_sock, reinterpret_cast<const char*>(&request_length), sizeof(uint32_t))) {
    AVA_LOG(FATAL) << "Fail to send request len to manager";
    abort();
  }
  if (!ava::support::SendData(manager_sock, reinterpret_cast<const char*>(request_buf.data()), request_length)) {
    AVA_LOG(FATAL) << "Fail to send request body to manager";
    abort();
  }

  // De-serialize API server addresses
  uint32_t reply_length = 0;
  if (!ava::support::RecvData(manager_sock, reinterpret_cast<char*>(&reply_length), sizeof(uint32_t), /* eof= */ nullptr)) {
    AVA_LOG(FATAL) << "Fail to receive reply len";
    abort();
  }

  std::vector<unsigned char> reply_buf(reply_length);
  zpp::serializer::memory_input_archive in(reply_buf);

  if (!ava::support::RecvData(manager_sock, reinterpret_cast<char*>(reply_buf.data()), reply_length, /* eof= */ nullptr)) {
    AVA_LOG(FATAL) << "Fail to receive reply from manager";
    abort();
  }
  ava_proto::WorkerAssignReply reply;
  in(reply);
  std::vector<std::string> worker_address;
  for (auto &wa : reply.worker_address()) {
    worker_address.push_back(wa);
  }
  if (worker_address.empty()) {
    AVA_ERROR << "No API server is assigned";
  }

  /* Connect API servers. */
  std::vector<struct command_channel *> channels;
  for (const auto &wa : worker_address) {
    /* Create a channel for every API server. */
    struct chansocketutil::command_channel_socket *chan =
        (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_tcp_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);
    channels.push_back((struct command_channel *)chan);

    char worker_name[128];
    int worker_port;
    struct hostent *worker_server_info;
    parseServerAddress(wa.c_str(), &worker_server_info, worker_name, &worker_port);
    assert(worker_server_info != NULL && "Unknown API server address");
    assert(worker_port > 0 && "Invalid API server port");
    AVA_INFO << "Assigned worker at " << worker_name << ":" << worker_port;

    chan->vm_id = nw_global_vm_id = 1;
    chan->listen_port = nw_worker_id = worker_port;

    /* Start a TCP client to connect API server at `worker_name:worker_port`. */
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr = *(struct in_addr *)worker_server_info->h_addr;
    address.sin_port = htons(worker_port);
    std::cerr << "Connect target API server (" << wa << ") at " << inet_ntoa(address.sin_addr) << ":" << worker_port
              << std::endl;

    int connect_ret = -1;
    auto connect_start = std::chrono::steady_clock::now();
    while (connect_ret) {
      chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);
      setsockopt_lowlatency(chan->sock_fd);
      connect_ret = connect(chan->sock_fd, (struct sockaddr *)&address, sizeof(address));
      if (!connect_ret) break;

      close(chan->sock_fd);
      auto connect_checkpoint = std::chrono::steady_clock::now();
      if ((uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(connect_checkpoint - connect_start).count() >
          guestconfig::config->connect_timeout_) {
        std::cerr << "Connection to " << worker_address[0] << " timeout" << std::endl;
        goto error;
      }
    }

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;
  }

  return channels;

error:
  for (auto &chan : channels) free(chan);
  channels.clear();
  return channels;
}

namespace {
struct command_channel_vtable command_channel_socket_tcp_vtable = {
    chansocketutil::command_channel_socket_buffer_size,      chansocketutil::command_channel_socket_new_command,
    chansocketutil::command_channel_socket_attach_buffer,    chansocketutil::command_channel_socket_send_command,
    chansocketutil::command_channel_socket_transfer_command, chansocketutil::command_channel_socket_receive_command,
    chansocketutil::command_channel_socket_get_buffer,       chansocketutil::command_channel_socket_get_data_region,
    chansocketutil::command_channel_socket_free_command,     chansocketutil::command_channel_socket_free,
    chansocketutil::command_channel_socket_print_command};
}
