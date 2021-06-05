#include "common/extensions/cmd_batching.h"

#include <gsl/gsl>

#include "common/linkage.h"

struct command_batch *nw_global_cmd_batch = NULL;  // always NULL

/**
 * API server side
 */

/**
 * __do_batch_execute - Execute received batch
 * @command_buffer: the buffer containing all batched commands
 * @total_buffer_size: total size of the batched commands
 *
 * We use AvA to generate the wrapper for this function, named __do_batch_emit.
 * TODO: we need to consider and process return values.
 */
EXPORTED_WEAKLY void __do_batch_execute(void *command_buffer, size_t total_buffer_size) {
  off_t offset = 0;
  struct command_base *cmd;

  while (gsl::narrow_cast<size_t>(offset) < total_buffer_size) {
    cmd = (struct command_base *)(command_buffer + offset);
    offset += cmd->command_size + cmd->region_size;
    __handle_command_cudart_opt_single(NULL, NULL, NULL, cmd);
  }
}

EXPORTED_WEAKLY void __do_batch_emit(void *command_buffer, size_t total_buffer_size) {
  __do_batch_execute(command_buffer, total_buffer_size);
}

/**
 * __handle_command_cudart_opt_single - Execute a single command
 * @chan: command channel
 * @handle_pool: object handle pool
 * @log: log channel
 * @cmd: command to be executed
 *
 * This function must be called once to set __chan and __handle_pool. Then it can be
 * called with only @cmd parameter to execute the command.
 */
EXPORTED_WEAKLY void __handle_command_cudart_opt_single(struct command_channel *chan,
                                                        struct nw_handle_pool *handle_pool, struct command_channel *log,
                                                        const struct command_base *cmd) {
  static struct command_channel *__chan = NULL;
  static struct nw_handle_pool *__handle_pool = NULL;
  static struct command_channel *__log = NULL;

  if (chan) {
    __chan = chan;
    __handle_pool = handle_pool;
    __log = log;
  }

  if (cmd) {
    assert(__chan && "Command channel has not been set");
#ifndef NDEBUG
    if (__print_command_onnx_opt)
      __print_command_onnx_opt(stderr, __chan, cmd);
    else if (__print_command_tf_opt)
      __print_command_tf_opt(stderr, __chan, cmd);
#endif
    if (__handle_command_onnx_opt)
      __handle_command_onnx_opt(__chan, __handle_pool, __log, cmd);
    else if (__handle_command_tf_opt)
      __handle_command_tf_opt(__chan, __handle_pool, __log, cmd);
  }
}

EXPORTED_WEAKLY struct command_batch *cmd_batch_thread_init(void) { return NULL; }

EXPORTED_WEAKLY void cmd_batch_thread_fini(struct command_batch *cmd_batch) {}
