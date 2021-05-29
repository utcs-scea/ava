#ifndef AVA_COMMON_CMD_CHANNEL_SOCKET_UTILITIES_HPP_
#define AVA_COMMON_CMD_CHANNEL_SOCKET_UTILITIES_HPP_

#include <poll.h>

#include <vector>

#include "cmd_channel_impl.hpp"

namespace chansocketutil {

struct command_channel_socket {
  struct command_channel_base base;
  int sock_fd;
  struct pollfd pfd;
  uint8_t vm_id;

  /* Channel locks */
  pthread_mutex_t send_mutex;
  pthread_mutex_t recv_mutex;

  // TODO: Remove the following fields that don't seem to do anything.
  int listen_fd;
  int listen_port;
  uint8_t init_command_type;

  std::vector<struct command_channel_socket *> channels;
};

void command_channel_socket_print_command(const struct command_channel *chan, const struct command_base *cmd);
void command_channel_socket_free(struct command_channel *c);
size_t command_channel_socket_buffer_size(const struct command_channel *c, size_t size);
struct command_base *command_channel_socket_new_command(struct command_channel *c, size_t command_struct_size,
                                                        size_t data_region_size);
void *command_channel_socket_attach_buffer(struct command_channel *c, struct command_base *cmd, void *buffer,
                                           size_t size);
void command_channel_socket_send_command(struct command_channel *c, struct command_base *cmd);
void command_channel_socket_transfer_command(struct command_channel *c, const struct command_channel *source,
                                             const struct command_base *cmd);
struct command_base *command_channel_socket_receive_command(struct command_channel *c);
void *command_channel_socket_get_buffer(const struct command_channel *chan, const struct command_base *cmd,
                                        void *buffer_id);
void *command_channel_socket_get_data_region(const struct command_channel *c, const struct command_base *cmd);
void command_channel_socket_free_command(struct command_channel *c, struct command_base *cmd);

}  // namespace chansocketutil

#endif  // AVA_COMMON_CMD_CHANNEL_SOCKET_UTILITIES_H_
