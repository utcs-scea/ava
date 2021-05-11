#ifndef AVA_COMMON_CMD_CHANNEL_HPP_
#define AVA_COMMON_CMD_CHANNEL_HPP_

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#include "common/socket.hpp"

#ifdef __cplusplus
#include <vector>

extern "C" {
#endif

/**
 * This API specifies how to send and receive commands over a
 * command_channel. It does not address how to open or close these
 * channels.
 *
 * Commands include both invocations and invocation results. So their
 * may well be more commands than API functions.
 *
 * This file defines an API and a shared data structure, but it does
 * NOT define an implementation. There could be any number of
 * different implementations of this API which use different
 * communication techniques and different memory layouts or allocation
 * techniques. As such, any statements about the memory layout of
 * buffers or the communication format of commands is out of scope for
 * this API. Specific implementations should make those decisions in
 * such a way that they can support the API specified here.
 */

struct command_channel;
struct command_channel_log;

/**
 * The "base struct" which must be at the beginning of every command
 * struct.
 *
 * The structure identifies the command and specifies sizes needed by
 * the command channel API. It also references the data region
 * associated with this command.
 *
 * All fields of this structure other than api_id and command_id are
 * internal to the command_channel implementation and should not be
 * modified or interpreted by other code. api_id and command_id should
 * be assigned by the sender and interpreted by the receiver.
 *
 * Commands do not carry the source VM because these are implied by
 * which command channel the command comes over.
 */
struct command_base {
  /**
   * The API ID as assigned in the API specification (for instance,
   * OpenCL might be 1 and CUDA might be 5). This identifies the
   * "namespace" in which `command_id` should be interpreted.
   */
  uint8_t api_id;
  /**
   * The VM ID is assigned by hypervisor.
   * FIXME: this should not be sent by guestlib but I have not found any
   * workaround.
   */
  uint8_t vm_id;
  /**
   * The type of the command.
   */
  uintptr_t command_type;
  /**
   * The ID of the thread which sent this command.
   */
  int64_t thread_id;
  int64_t original_thread_id;
  /**
   * The status of the command execution.
   * TODO: it is defined in include/socket.h but commented.
   */
  /**
   * The flags of the command, and is assigned by hypervisor to mark the
   * status of the command.
   */
  int8_t flags;
  /**
   * The command ID (within the API). This ID defines what fields this
   * command struct contains.
   */
  uintptr_t command_id;
  /**
   * The size of this command struct.
   */
  size_t command_size;
  /**
   * A reference to the data region associated with this command. It
   * may be a pointer, but can also be an offset or something else.
   */
  void *data_region;
  /**
   * The size of the data region attached to this command.
   */
  size_t region_size;
  /**
   * Reserved region for other purposes, for example, param_block seeker
   * in shared memory implementation. */
  char reserved_area[64];
};

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
void command_channel_free(struct command_channel *c);

//! Sending

/**
 * Compute the buffer size that will actually be used for a buffer of
 * `size`. The returned value may be larger than `size`.
 */
__attribute__((pure)) size_t command_channel_buffer_size(const struct command_channel *chan, size_t size);

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
__attribute__((malloc)) struct command_base *command_channel_new_command(struct command_channel *chan,
                                                                         size_t command_struct_size,
                                                                         size_t data_region_size);

/**
 * Attach a buffer to a command and return a location independent
 * buffer ID. `buffer` must be valid until after the call to
 * `command_channel_send_command` or NULL.
 *
 * The combined attached buffers must fit within the initially
 * provided `data_region_size` (to `command_channel_new_command`).
 */
void *command_channel_attach_buffer(struct command_channel *chan, struct command_base *cmd, const void *buffer,
                                    size_t size);

/**
 * Send the message and all its attached buffers.
 *
 * This call is asynchronous and does not block for the command to
 * complete execution.
 */
void command_channel_send_command(struct command_channel *chan, struct command_base *cmd);

/**
 * Take a command received (or loaded) in one channel and send it along another.
 *
 * This call is asynchronous and does not block for the command to
 * complete execution.
 *
 * @param chan The output channel.
 * @param source_chan The input channel.
 * @param cmd The command received on `source_chan`.
 */
void command_channel_transfer_command(struct command_channel *chan, const struct command_channel *source_chan,
                                      const struct command_base *cmd);

//! Receiving

/**
 * Receive a command from a channel. The returned Command pointer
 * should be interpreted based on its `command_id` field.
 *
 * This call blocks waiting for a command to be sent along this
 * channel.
 */
__attribute__((malloc)) struct command_base *command_channel_receive_command(struct command_channel *chan);

/**
 * Translate a buffer_id (as returned by
 * `command_channel_attach_buffer` in the sender) into a data pointer.
 * The returned pointer will be valid until
 * `command_channel_free_command` is called on `cmd`.
 */
__attribute__((pure)) void *command_channel_get_buffer(const struct command_channel *chan,
                                                       const struct command_base *cmd, const void *buffer_id);

/**
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
__attribute__((pure)) void *command_channel_get_data_region(const struct command_channel *c,
                                                            const struct command_base *cmd);

/**
 * Free a command returned by `command_channel_receive_command`.
 */
void command_channel_free_command(struct command_channel *chan, struct command_base *cmd);

/**
 * Print a command for debugging.
 */
void command_channel_print_command(const struct command_channel *chan, const struct command_base *cmd);

//! Examples

//! Shared Pseudocode

/*

success(0)
char simple_func(int size, buffer(size) int* buffer, ... and more ...);

enum {
  CMD_SIMPLE_FUNC_CALL,
  CMD_SIMPLE_FUNC_RET,
  ...
};

struct simple_func_call {
  struct command_base base;

  int size;
  int* buffer;
  ...
};

struct simple_func_return {
  struct command_base base;

  int* buffer;
  ...
  char ret;
};

*/

//! Guestlib Pseudocode

/*

struct command_channel* chan;

char simple_func(int size, int* buffer, ... and more ...)
{
  size_t total_buffer_size = 0;
  total_buffer_size += command_channel_buffer_size(chan, size * sizeof(int));
  ... add sizes of more buffers (including potentially nested buffers) ...;

  simple_func_call *call = (simple_func_call *)command_channel_new_command(chan, CMD_SIMPLE_FUNC_CALL,
sizeof(simple_func_call), total_buffer_size);

  call->size = size;
  call->buffer = command_channel_attach_buffer(chan, call, buffer, size * sizeof(int));
  ... assign more values into call and attach additional buffers ...;

  command_channel_send_command(chan, call);

  if (simple_func is synchronous) {
    simple_func_return *ret = (simple_func_return *)command_channel_receive_command(chan);
    assert(ret->command_id == CMD_SIMPLE_FUNC_RET);
    // Imagining buffer is in/out
    int* buffer_ret = command_channel_get_buffer(chan, ret, ret->buffer);
    memcpy(buffer, buffer_ret, size * sizeof(int));
    ... Copy back other out buffers ...;
    command_channel_free_command(chan, ret);
    return ret->ret;
  } else if(simple_func is asynchronous) {
    // Imagining buffer is in only
    return 0; // Success
  }
}

*/

//! API Server Pseudocode

/*

void command_execution_loop() {
  while(1) {
    struct command_base* cmd = command_channel_receive_command(chan);
    switch(cmd->command_id) {
    case CMD_SIMPLE_FUNC_CALL:
      simple_func_call *call = (simple_func_call *)cmd;
      int size = call->size;
      int* buffer = command_channel_get_buffer(chan, ret, call->buffer);
      ... convert more arguments ...;

      char ret = simple_func(size, buffer);

      if (simple_func is synchronous) {
        size_t total_buffer_size = 0;
        total_buffer_size += command_channel_buffer_size(chan, size * sizeof(int));
        ... add sizes of more buffers (including potentially nested buffers) ...;

        simple_func_return *ret_cmd = (simple_func_return *)command_channel_new_command(chan, CMD_SIMPLE_FUNC_RET,
sizeof(SimpleFuncRet), total_buffer_size);
        // Imagining buffer is in/out
        ret_cmd->buffer = command_channel_attach_buffer(chan, ret_cmd, buffer, size * sizeof(int));
        ... assign more values into call and attach additional buffers ...;
        ret_cmd->ret = ret;
        command_channel_send_command(chan, ret_cmd);
      } else if(simple_func is asynchronous) {
        // Imagining buffer is in only
        // No return because we are async.
      }
      command_channel_free_command(chan, call);
      break;
    }
  };
}

*/

//! Channel constructors
struct command_channel *command_channel_socket_new(void);
struct command_channel *command_channel_socket_worker_new(int listen_port);
struct command_channel *command_channel_socket_tcp_migration_new(int worker_port, int is_source);
struct command_channel_log *command_channel_log_new(int worker_port);

//! Hypervisor

struct command_channel *command_channel_hv_new(int worker_port);
void command_channel_hv_free(struct command_channel *c);

/**
 * Report resource usages to the hypervisor.
 */
void command_channel_hv_report_storage_resource_allocation(struct command_channel *c, const char *const name,
                                                           ssize_t amount);
void command_channel_hv_report_throughput_resource_consumption(struct command_channel *c, const char *const name,
                                                               ssize_t amount);

//! Configurations

enum command_channel_type { AVA_LOCAL, AVA_SHM, AVA_VSOCK };

//! Log channels

ssize_t command_channel_log_transfer_command(struct command_channel_log *c, const struct command_channel *source,
                                             const struct command_base *cmd);
void command_channel_log_update_flags(struct command_channel_log *chan, ssize_t offset, uint32_t flags);
struct command_base *command_channel_log_load_command(struct command_channel_log *c, ssize_t offset, uint32_t *flags);

#ifdef __cplusplus
}
#endif

#endif  // AVA_COMMON_CMD_CHANNEL_HPP_
