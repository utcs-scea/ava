/**
 * This file implements the command batching optimization for TensorFlow 1.14 and
 * ONNXruntime 1.2.0.
 * The underlying dependencies are CUDA 10.1 and cuDNN 7.6.5.
 * The optimization is applied in `cava/samples/onnxruntime/onnx_opt.c`.
 */
#ifndef AVA_EXTENSIONS_CMD_BATCHING_H_
#define AVA_EXTENSIONS_CMD_BATCHING_H_

#include <assert.h>
#include <glib.h>
#include <stdio.h>

#ifndef __CAVA__
#include "cmd_channel.hpp"
#include "cmd_handler.hpp"
#include "endpoint_lib.hpp"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Batch parameters
 *
 * face: 15, 10000, 5000
 * tts: 500, 10000, 2000
 * onnx_arcface: 50, 10000, 5000
 */
#define BATCH_SIZE 50
#define BATCH_TIME_OUT_US 10000
#define BATCH_QUEUE_TIME_OUT_US 5000

struct command_batch {
  GAsyncQueue *pending_cmds;
  GAsyncQueue *active_cmds;
  pthread_t process_thread;
  int guest_stats_fd;
  int running;
};

extern struct command_batch *nw_global_cmd_batch;

struct command_batch *cmd_batch_thread_init(void);
void cmd_batch_thread_fini(struct command_batch *cmd_batch);
void batch_insert_command(struct command_batch *cmd_batch, struct command_base *cmd, struct command_channel *chan,
                          int is_async);

void __do_batch_emit(void *command_buffer, size_t total_buffer_size);
void __do_batch_execute(void *command_buffer, size_t total_buffer_size);

void __handle_command_cudart_opt_single(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                                        struct command_channel *log, const struct command_base *cmd);

void __handle_command_onnx_opt(struct command_channel *__chan, struct nw_handle_pool *handle_pool,
                               struct command_channel *__log, const struct command_base *__cmd) __attribute__((weak));
void __print_command_onnx_opt(FILE *file, const struct command_channel *__chan, const struct command_base *__cmd)
    __attribute__((weak));

void __handle_command_tf_opt(struct command_channel *__chan, struct nw_handle_pool *handle_pool,
                             struct command_channel *__log, const struct command_base *__cmd) __attribute__((weak));
void __print_command_tf_opt(FILE *file, const struct command_channel *__chan, const struct command_base *__cmd)
    __attribute__((weak));

#ifdef __cplusplus
}
#endif

#endif  // AVA_EXTENSIONS_CMD_BATCHING_H_
