#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "cmd_channel_socket_utilities.hpp"
#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/devconf.h"
#include "common/guest_mem.h"
#include "common/logging.h"
#include "manager_service.proto.h"

extern int nw_global_vm_id;

namespace {
extern struct command_channel_vtable command_channel_socket_tcp_vtable;
}

struct command_channel *command_channel_socket_tcp_migration_new(int worker_port, int is_source) {
  struct chansocketutil::command_channel_socket *chan =
      (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
  command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_tcp_vtable);
  pthread_mutex_init(&chan->send_mutex, NULL);
  pthread_mutex_init(&chan->recv_mutex, NULL);

  chan->listen_port = worker_port + 2000;

  struct sockaddr_in address;
  int addrlen = sizeof(address);
  int opt;
  memset(&address, 0, sizeof(address));

  if (is_source) {
    /* start TCP client */
    chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);

    address.sin_family = AF_INET;
    address.sin_port = htons(chan->listen_port);
    inet_pton(AF_INET, DEST_SERVER_IP, &address.sin_addr);
    printf("connect target worker@%s:%d\n", DEST_SERVER_IP, chan->listen_port);
    connect(chan->sock_fd, (struct sockaddr *)&address, sizeof(address));
  } else {
    /* start TCP server */
    if ((chan->listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
      perror("socket");
    }
    // Forcefully attaching socket to the worker port
    opt = 1;
    if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
      perror("setsockopt reuseaddr");
    }
    opt = 1;
    if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt))) {
      perror("setsockopt reuseport");
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
#ifndef NDEBUG
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
    chansocketutil::command_channel_socket_buffer_size,      chansocketutil::command_channel_socket_new_command,
    chansocketutil::command_channel_socket_attach_buffer,    chansocketutil::command_channel_socket_send_command,
    chansocketutil::command_channel_socket_transfer_command, chansocketutil::command_channel_socket_receive_command,
    chansocketutil::command_channel_socket_get_buffer,       chansocketutil::command_channel_socket_get_data_region,
    chansocketutil::command_channel_socket_free_command,     chansocketutil::command_channel_socket_free,
    chansocketutil::command_channel_socket_print_command};
}

// warning TODO: Does there need to be a separate socket specific function which
// handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the
// chansocketutil::command_channel_socket_new and other socket specific APIs.
