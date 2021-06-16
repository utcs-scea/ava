#include "common/cmd_channel_impl.hpp"
#include "common/endpoint_lib.hpp"

// TODO: actually not used but can't get rid of it...
int nw_worker_id = 0;
int nw_global_vm_id = 1;

void command_channel_free(struct command_channel *chan) {
  ((struct command_channel_base *)chan)->vtable->command_channel_free(chan);
}

size_t command_channel_buffer_size(const struct command_channel *chan, size_t size) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_buffer_size(chan, size);
}

struct command_base *command_channel_new_command(struct command_channel *chan, size_t command_struct_size,
                                                 size_t data_region_size) {
  return ((struct command_channel_base *)chan)
      ->vtable->command_channel_new_command(chan, command_struct_size, data_region_size);
}

void *command_channel_attach_buffer(struct command_channel *chan, struct command_base *cmd, const void *buffer,
                                    size_t size) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_attach_buffer(chan, cmd, (void *)buffer, size);
}

void command_channel_send_command(struct command_channel *chan, struct command_base *cmd) {
  ((struct command_channel_base *)chan)->vtable->command_channel_send_command(chan, cmd);
}

void command_channel_transfer_command(struct command_channel *chan, const struct command_channel *source_chan,
                                      const struct command_base *cmd) {
  ((struct command_channel_base *)chan)->vtable->command_channel_transfer_command(chan, source_chan, cmd);
}

struct command_base *command_channel_receive_command(struct command_channel *chan) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_receive_command(chan);
}

void *command_channel_get_buffer(const struct command_channel *chan, const struct command_base *cmd,
                                 const void *buffer_id) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_get_buffer(chan, cmd, (void *)buffer_id);
}

void *command_channel_get_data_region(const struct command_channel *chan, const struct command_base *cmd) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_get_data_region(chan, cmd);
}

void command_channel_free_command(struct command_channel *chan, struct command_base *cmd) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_free_command(chan, cmd);
}

void command_channel_print_command(const struct command_channel *chan, const struct command_base *cmd) {
  return ((struct command_channel_base *)chan)->vtable->command_channel_print_command(chan, cmd);
}

void command_channel_simple_print_command(const struct command_channel *AVA_UNUSED(chan),
                                          const struct command_base *AVA_UNUSED(cmd)) {
  DEBUG_PRINT_COMMAND(chan, cmd);
}
