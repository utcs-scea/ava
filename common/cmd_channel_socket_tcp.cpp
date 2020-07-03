#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"
#include "common/guest_mem.h"
#include "common/cmd_handler.h"
#include "cmd_channel_socket_utilities.h"
#include "guest_config.h"
#include "manager_service.h"

extern int nw_global_vm_id;

namespace {
  extern struct command_channel_vtable command_channel_socket_tcp_vtable;
}

/**
 * TCP channel guestlib endpoint.
 *
 * The `manager_tcp` is required to use the TCP channel.
 * */
std::vector<struct command_channel*> command_channel_socket_tcp_guest_new()
{
    auto channel = grpc::CreateChannel(guestconfig::config->manager_address_, grpc::InsecureChannelCredentials());
    auto client  = std::make_unique<ManagerServiceClient>(channel);
    std::vector<uint64_t> gpu_mem_in_bytes;
    for (auto m : guestconfig::config->gpu_memory_)
      gpu_mem_in_bytes.push_back(m << 20);
    std::vector<std::string> worker_address =
        client->AssignWorker(0, guestconfig::config->gpu_memory_.size(), gpu_mem_in_bytes);
    assert(!worker_address.empty() && "No API server is assigned");

    /* Connect API servers. */
    std::vector<struct command_channel*> channels;
    for (const auto& wa : worker_address) {
      /* Create a channel for every API server. */
      struct chansocketutil::command_channel_socket* chan =
        (struct chansocketutil::command_channel_socket*)malloc(sizeof(struct chansocketutil::command_channel_socket));
      command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_tcp_vtable);
      pthread_mutex_init(&chan->send_mutex, NULL);
      pthread_mutex_init(&chan->recv_mutex, NULL);
      channels.push_back((struct command_channel*)chan);

      char worker_name[128];
      int worker_port;
      struct hostent *worker_server_info;
      parseServerAddress(wa.c_str(), &worker_server_info, worker_name, &worker_port);
      assert(worker_server_info != NULL && "Unknown API server address");
      assert(worker_port > 0 && "Invalid API server port");
      DEBUG_PRINT("Assigned worker at %s:%d\n", worker_name, worker_port);

      chan->vm_id = nw_global_vm_id = 1;
      chan->listen_port = nw_worker_id = worker_port;

      /* Start a TCP client to connect API server at `worker_name:worker_port`. */
      struct sockaddr_in address;
      memset(&address, 0, sizeof(address));
      address.sin_family = AF_INET;
      address.sin_addr = *(struct in_addr *)worker_server_info->h_addr;
      address.sin_port = htons(worker_port);
      std::cerr <<  "Connect target API server (" << wa << ") at "
                << inet_ntoa(address.sin_addr) << ":" << worker_port << std::endl;

      int connect_ret = -1;
      auto connect_start = std::chrono::steady_clock::now();
      while (connect_ret) {
        chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt_lowlatency(chan->sock_fd);
        connect_ret = connect(chan->sock_fd, (struct sockaddr *)&address, sizeof(address));
        if (!connect_ret)
          break;

        close(chan->sock_fd);
        auto connect_checkpoint = std::chrono::steady_clock::now();
        if ((uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
              connect_checkpoint - connect_start).count() > guestconfig::config->connect_timeout_) {
          std::cerr << "Connection to " << worker_address[0] << " timeout" << std::endl;
          goto error;
        }
      }

      chan->pfd.fd = chan->sock_fd;
      chan->pfd.events = POLLIN | POLLRDHUP;
    }

    return channels;

error:
    for (auto& chan : channels)
      free(chan);
    channels.clear();
    return channels;
}

/**
 * TCP channel API server endpoint.
 * @worker_port: the listening port of worker.
 *
 * The `manager_tcp` is required to use the TCP channel.
 */
struct command_channel* command_channel_socket_tcp_worker_new(int worker_port)
{
    struct chansocketutil::command_channel_socket *chan = (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_tcp_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    memset(&address, 0, sizeof(address));

    chan->listen_port = worker_port;

    /* start TCP server */
    if ((chan->listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket");
    }
    // Forcefully attaching socket to the worker port
    if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(chan->listen_port);

    if (bind(chan->listen_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
    }
    if (listen(chan->listen_fd, 10) < 0) {
        perror("listen");
    }

    fprintf(stderr, "[%d] Waiting for guestlib connection\n", chan->listen_port);
    chan->sock_fd = accept(chan->listen_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
    if (chan->sock_fd < 0) {
       perror("accept");
    }
    setsockopt_lowlatency(chan->sock_fd);

    /* Get source address */
#ifdef DEBUG
    struct sockaddr_storage source_addr;
    socklen_t source_addr_len = sizeof(struct sockaddr_storage);
    getpeername(chan->sock_fd, (struct sockaddr *)&source_addr, &source_addr_len);
    if (source_addr.ss_family == AF_INET) {
        struct sockaddr_in *s = (struct sockaddr_in *)&source_addr;
        char ipstr[64];
        inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof(ipstr));
        fprintf(stderr, "[%d] Accept guestlib at %s:%d\n", chan->listen_fd, ipstr, ntohs(s->sin_port));
    }
#endif

    /* Receive handler initialization API */
    struct command_handler_initialize_api_command init_msg;
    recv_socket(chan->sock_fd, &init_msg, sizeof(struct command_handler_initialize_api_command));
    chan->init_command_type = init_msg.new_api_id;
    chan->vm_id = init_msg.base.vm_id;
    fprintf(stderr, "[%d] Accept guestlib with API_ID=%x\n",
            chan->listen_port, chan->init_command_type);

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

struct command_channel* command_channel_socket_tcp_migration_new(int worker_port, int is_source)
{
    struct chansocketutil::command_channel_socket *chan =
      (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_tcp_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    chan->listen_port = worker_port + 2000;

    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    memset(&address, 0, sizeof(address));

    if (is_source) {
        /* start TCP client */
        chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);

        address.sin_family = AF_INET;
        address.sin_port = htons(chan->listen_port);
        inet_pton(AF_INET, DEST_SERVER_IP, &address.sin_addr);
        printf("connect target worker@%s:%d\n", DEST_SERVER_IP, chan->listen_port);
        connect(chan->sock_fd, (struct sockaddr *)&address, sizeof(address));
    }
    else {
        /* start TCP server */
        if ((chan->listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
            perror("socket");
        }
        // Forcefully attaching socket to the worker port
        if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
            perror("setsockopt");
        }
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(chan->listen_port);

        if (bind(chan->listen_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
            perror("bind failed");
        }
        if (listen(chan->listen_fd, 10) < 0) {
            perror("listen");
        }

        printf("[target worker@%d] waiting for source worker connection\n", chan->listen_port);
        chan->sock_fd = accept(chan->listen_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);

        /* Get source address */
#ifdef DEBUG
        struct sockaddr_storage source_addr;
        socklen_t source_addr_len = sizeof(struct sockaddr_storage);
        getpeername(chan->sock_fd, (struct sockaddr *)&source_addr, &source_addr_len);
        if (source_addr.ss_family == AF_INET) {
            struct sockaddr_in *s = (struct sockaddr_in *)&source_addr;
            char ipstr[64];
            inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof(ipstr));
            printf("accept source worker@%s:%d\n", ipstr, ntohs(s->sin_port));
        }
#endif
    }

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

namespace {
  struct command_channel_vtable command_channel_socket_tcp_vtable = {
    chansocketutil::command_channel_socket_buffer_size,
    chansocketutil::command_channel_socket_new_command,
    chansocketutil::command_channel_socket_attach_buffer,
    chansocketutil::command_channel_socket_send_command,
    chansocketutil::command_channel_socket_transfer_command,
    chansocketutil::command_channel_socket_receive_command,
    chansocketutil::command_channel_socket_get_buffer,
    chansocketutil::command_channel_socket_get_data_region,
    chansocketutil::command_channel_socket_free_command,
    chansocketutil::command_channel_socket_free,
    chansocketutil::command_channel_socket_print_command
  };
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the chansocketutil::command_channel_socket_new and other socket specific APIs.
