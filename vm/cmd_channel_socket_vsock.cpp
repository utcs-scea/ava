#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <plog/Log.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_channel_socket_utilities.hpp"
#include "common/cmd_handler.hpp"
#include "common/debug.hpp"
#include "common/devconf.h"
#include "common/guest_mem.h"

extern int nw_global_vm_id;

namespace {
extern struct command_channel_vtable command_channel_socket_vsock_vtable;
}

struct command_channel *command_channel_socket_new() {
  struct chansocketutil::command_channel_socket *chan =
      (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
  command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vsock_vtable);
  pthread_mutex_init(&chan->send_mutex, NULL);
  pthread_mutex_init(&chan->recv_mutex, NULL);

  chan->vm_id = nw_global_vm_id = 1;

  /**
   * Get manager's host address from ENV('AVA_MANAGER_ADDR').
   * The address can either be a full IP:port (e.g. 0.0.0.0:3333),
   * or only the port (3333), but the IP address is always ignored as
   * the manager is assumed to be on the local server.
   */
  char *manager_full_address;
  int manager_port;
  manager_full_address = getenv("AVA_MANAGER_ADDR");
  assert(manager_full_address != NULL && "AVA_MANAGER_ADDR is not set");
  parseServerAddress(manager_full_address, NULL, NULL, &manager_port);
  assert(manager_port > 0 && "Invalid manager port");

  /* connect manager to get worker port */
  struct sockaddr_vm sa;
  int manager_fd = init_vm_socket(&sa, VMADDR_CID_HOST, manager_port);
  conn_vm_socket(manager_fd, &sa);

  struct command_base *msg = chansocketutil::command_channel_socket_new_command((struct command_channel *)chan,
                                                                                sizeof(struct command_base), 0);
  msg->command_type = NW_NEW_APPLICATION;
  send_socket(manager_fd, msg, sizeof(struct command_base));

  recv_socket(manager_fd, msg, sizeof(struct command_base));
  uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
  assert(nw_worker_id == 0);  // TODO: Move assignment to nw_worker_id out of
                              // unrelated constructor.
  nw_worker_id = worker_port;
  chansocketutil::command_channel_socket_free_command((struct command_channel *)chan, msg);
  close(manager_fd);

  /* connect worker */
  LOG_INFO << "assigned worker at " << worker_port;
  chan->sock_fd = init_vm_socket(&sa, VMADDR_CID_HOST, worker_port);
  // FIXME: connect is always non-blocking for vm socket!
  if (!getenv("AVA_WPOOL") || !strcmp(getenv("AVA_WPOOL"), "FALSE")) usleep(2000000);
  conn_vm_socket(chan->sock_fd, &sa);

  chan->pfd.fd = chan->sock_fd;
  chan->pfd.events = POLLIN | POLLRDHUP;

  return (struct command_channel *)chan;
}

struct command_channel *command_channel_socket_worker_new(int listen_port) {
  struct chansocketutil::command_channel_socket *chan =
      (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
  command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vsock_vtable);
  pthread_mutex_init(&chan->send_mutex, NULL);
  pthread_mutex_init(&chan->recv_mutex, NULL);

  // TODO: notify executor when VM created or destroyed
  printf("spawn worker port#%d\n", listen_port);
  chan->listen_port = listen_port;
  assert(nw_worker_id == 0);  // TODO: Move assignment to nw_worker_id out of
                              // unrelated constructor.
  nw_worker_id = listen_port;

  /* connect guestlib */
  struct sockaddr_vm sa_listen;
  chan->listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, chan->listen_port);
  listen_vm_socket(chan->listen_fd, &sa_listen);

  printf("[worker@%d] waiting for guestlib connection\n", listen_port);
  chan->sock_fd = accept_vm_socket(chan->listen_fd, NULL);

  struct command_handler_initialize_api_command init_msg;
  recv_socket(chan->sock_fd, &init_msg, sizeof(struct command_handler_initialize_api_command));
  chan->init_command_type = init_msg.new_api_id;
  chan->vm_id = init_msg.base.vm_id;
  printf("[worker@%d] vm_id=%d, api_id=%x\n", listen_port, chan->vm_id, chan->init_command_type);

  // TODO: also poll netlink socket, and put the swapping task in the same
  // task queue just as the normal invocations.
  chan->pfd.fd = chan->sock_fd;
  chan->pfd.events = POLLIN | POLLRDHUP;

  /*
  if (fcntl(ex_st.client_fd, F_SETFL,
            fcntl(ex_st.client_fd, F_GETFL) & (~O_NONBLOCK)) < 0) {
      perror("fcntl blocking failed");
      return 0;
  }
  */

  return (struct command_channel *)chan;
}

namespace {
struct command_channel_vtable command_channel_socket_vsock_vtable = {
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
