#ifndef AVA_COMMON_CMD_CHANNEL_IMPL_HPP_
#define AVA_COMMON_CMD_CHANNEL_IMPL_HPP_

#ifndef __KERNEL__

#include <assert.h>

#include "cmd_channel.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct command_channel_vtable {
  size_t (*command_channel_buffer_size)(const struct command_channel *chan, size_t size);
  struct command_base *(*command_channel_new_command)(struct command_channel *chan, size_t command_struct_size,
                                                      size_t data_region_size);
  void *(*command_channel_attach_buffer)(struct command_channel *chan, struct command_base *cmd, void *buffer,
                                         size_t size);
  void (*command_channel_send_command)(struct command_channel *chan, struct command_base *cmd);
  void (*command_channel_transfer_command)(struct command_channel *chan, const struct command_channel *target,
                                           const struct command_base *cmd);
  struct command_base *(*command_channel_receive_command)(struct command_channel *chan);
  void *(*command_channel_get_buffer)(const struct command_channel *chan, const struct command_base *cmd,
                                      void *buffer_id);
  void *(*command_channel_get_data_region)(const struct command_channel *c, const struct command_base *cmd);
  void (*command_channel_free_command)(struct command_channel *chan, struct command_base *cmd);
  void (*command_channel_free)(struct command_channel *chan);
  void (*command_channel_print_command)(const struct command_channel *chan, const struct command_base *cmd);
};

#define __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, n) \
  assert(vtable.n != NULL && (#vtable " is missing value for " #n))
#define COMMAND_CHANNEL_VTABLE_CHECK(vtable)                                       \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_buffer_size);      \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_new_command);      \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_attach_buffer);    \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_send_command);     \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_transfer_command); \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_receive_command);  \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_get_buffer);       \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_get_data_region);  \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_free_command);     \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_free);             \
  __COMMAND_CHANNEL_VTABLE_CHECK_METHOD(vtable, command_channel_print_command)

/**
 * The "base" structure for command channels, this must be the first field of every command channel.
 */
struct command_channel_base {
  struct command_channel_vtable *vtable;
};

static inline void command_channel_preinitialize(struct command_channel *chan, struct command_channel_vtable *vtable) {
  ((struct command_channel_base *)chan)->vtable = vtable;
  COMMAND_CHANNEL_VTABLE_CHECK((*vtable));
}

/// A simple default implementation of print_command for use in command_channel implementations.
void command_channel_simple_print_command(const struct command_channel *chan, const struct command_base *cmd);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif

#endif  // AVA_COMMON_CMD_CHANNEL_IMPL_HPP_
