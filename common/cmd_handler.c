#include "common/linkage.h"
#include "common/debug.h"
#include "common/cmd_handler.h"
#include "common/endpoint_lib.h"
#include "common/shadow_thread_pool.h"

#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

// Internal flag set by the first call to init_command_handler
EXPORTED_WEAKLY volatile int init_command_handler_executed;

EXPORTED_WEAKLY struct command_channel* nw_global_command_channel;
EXPORTED_WEAKLY pthread_mutex_t nw_handler_lock = PTHREAD_MUTEX_INITIALIZER;

struct command_handler_t {
    void (*replay)(struct command_channel *__chan, struct nw_handle_pool *handle_pool,
                   struct command_channel* __log,
                   const struct command_base *__call_cmd, const struct command_base *__ret_cmd);
    void (*handle)(struct command_channel *__chan, struct nw_handle_pool *handle_pool,
                   struct command_channel *__log, const struct command_base *__cmd);
    void (*print)(FILE *file, const struct command_channel *__chan, const struct command_base *__cmd);
};

EXPORTED_WEAKLY struct command_handler_t nw_apis[MAX_API_ID];
EXPORTED_WEAKLY pthread_t nw_handler_thread;

static int handle_command(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                           struct command_channel *log, struct command_base *cmd);

EXPORTED_WEAKLY void register_command_handler(
        int api_id,
        void (*handle)(struct command_channel *, struct nw_handle_pool *,
                       struct command_channel *, const struct command_base *),
        void (*print)(FILE *, const struct command_channel *, const struct command_base *),
        void (*replay)(struct command_channel *, struct nw_handle_pool *,
                       struct command_channel* ,
                       const struct command_base *, const struct command_base *))
{
    assert(api_id < MAX_API_ID);
    DEBUG_PRINT("Registering API command handler for API id %d: handler at 0x%lx\n", api_id, (uintptr_t)handle);
    struct command_handler_t *api = &nw_apis[api_id];
    assert(api->handle == NULL && "Only one handler can be registered for each API id");
    api->handle = handle;
    api->print = print;
    api->replay = replay;
}

EXPORTED_WEAKLY void print_command(FILE* file, const struct command_channel *chan, const struct command_base *cmd) {
    const intptr_t api_id = cmd->api_id;
    // Lock the file to prevent commands from getting mixed in the print out
    flockfile(file);
    if (nw_apis[api_id].print)
        nw_apis[api_id].print(file, chan, cmd);
    funlockfile(file);
}


static int handle_command(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                           struct command_channel *log, struct command_base *cmd) {
    const intptr_t api_id = cmd->api_id;
    assert(nw_apis[api_id].handle != NULL);
    nw_apis[api_id].handle(chan, handle_pool, log, cmd);
    command_channel_free_command(chan, cmd);
    return 0;
}

static void _handle_commands_loop(struct command_channel* chan) {
    while(1) {
        struct command_base* cmd = command_channel_receive_command(chan);

#ifdef AVA_PRINT_TIMESTAMP
        if (cmd->api_id != 0) {
            struct timeval ts;
            gettimeofday(&ts, NULL);
            printf("Handler: command_%ld receive_command at : %ld s, %ld us\n", cmd->command_id, ts.tv_sec, ts.tv_usec);
        }
#endif

        // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.
        shadow_thread_pool_dispatch(nw_shadow_thread_pool, chan, cmd);
    }
}

void handle_command_and_notify(struct command_channel *chan, struct command_base *cmd)
{
    handle_command(chan, nw_global_handle_pool,
                                   (struct command_channel *) nw_record_command_channel, cmd);
}

static void* dispatch_thread_impl(void* userdata) {
    struct command_channel* chan = (struct command_channel*)userdata;

    // set cancellation state
    if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL)) {
        perror("pthread_setcancelstate failed\n");
        exit(0);
    }

    // PTHREAD_CANCEL_DEFERRED means that it will wait the pthread_join
    if (pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL)) {
        perror("pthread_setcanceltype failed\n");
        exit(0);
    }

    _handle_commands_loop(chan);
    return NULL;
}
// TODO: This will not correctly handle running callbacks in the initially calling thread.

EXPORTED_WEAKLY void init_command_handler(struct command_channel* (*channel_create)()) {
    pthread_mutex_lock(&nw_handler_lock);
    if (!init_command_handler_executed) {
        nw_global_command_channel = channel_create();
        pthread_create(&nw_handler_thread, NULL,
                       dispatch_thread_impl, (void*)nw_global_command_channel);
        atomic_thread_fence(memory_order_release);
        init_command_handler_executed = 1;
    }
    pthread_mutex_unlock(&nw_handler_lock);
}

EXPORTED_WEAKLY void destroy_command_handler() {
    pthread_mutex_lock(&nw_handler_lock);
    if (init_command_handler_executed) {
        pthread_cancel(nw_handler_thread);
        pthread_join(nw_handler_thread, NULL);
        command_channel_free(nw_global_command_channel);
        atomic_thread_fence(memory_order_release);
        init_command_handler_executed = 0;
    }
    pthread_mutex_unlock(&nw_handler_lock);
}

EXPORTED_WEAKLY void wait_for_command_handler() {
    pthread_join(nw_handler_thread, NULL);
}

//! Feel free to move these functions around

EXPORTED_WEAKLY struct command_channel_log* nw_record_command_channel;

static void replay_command(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                           struct command_channel* log,
                           const struct command_base *call_cmd, const struct command_base *ret_cmd) {
    const intptr_t api_id = call_cmd->api_id;
    assert(call_cmd->api_id == ret_cmd->api_id);
    assert(nw_apis[api_id].replay != NULL);
    nw_apis[api_id].replay(chan, handle_pool, log, call_cmd, ret_cmd);
}


// TODO: Should be removed. See COMMAND_END_MIGRATION below.
// Flag used to communicate with the guestlib. It is extern declared again at guestlib/src/init.c:102
EXPORTED_WEAKLY int nw_end_migration_flag = 0;

void internal_api_handler(struct command_channel *chan, struct nw_handle_pool *handle_pool, struct command_channel *log,
                          const struct command_base *cmd) {
    assert(cmd->api_id == COMMAND_HANDLER_API);

    struct command_channel *transfer_chan;

    switch (cmd->command_id) {
        case COMMAND_HANDLER_SHUTDOWN_API:
            exit(0);
            break;

        /**
         * For testing, guestlib initiates the migration and worker
         * replays the logs that recorded by itself.
         */
        case COMMAND_START_MIGRATION:
            //! Complete steps
            // Create a log channel for sending;
            // (Spawn a new worker which) Creates a log channel for receiving;
            // Transfer logs from nw_record_command_channel to sending channel by
            // ava_extract_objects(sending_chan, nw_record_command_channel,
            //         nw_handle_pool_get_live_handles(nw_global_handle_pool));
            // New worker executes received logs.

            //! Simplified steps
            // Create a log channel for sending and receiving
            transfer_chan = (struct command_channel *)command_channel_log_new(nw_worker_id + 1000);

            // Transfer logs from nw_record_command_channel to new log channel
            ava_extract_objects(transfer_chan, nw_record_command_channel,
                    nw_handle_pool_get_live_handles(nw_global_handle_pool));
            {
                struct command_base *log_end = command_channel_new_command(transfer_chan, sizeof(struct command_base),
                                                                           0);
                log_end->api_id = COMMAND_HANDLER_API;
                log_end->command_id = COMMAND_END_MIGRATION;
                command_channel_send_command(transfer_chan, log_end);
            }

            command_channel_free((struct command_channel *) nw_record_command_channel);

            struct nw_handle_pool *replay_handle_pool = nw_handle_pool_new();
            struct command_channel_log *replay_log = command_channel_log_new(nw_worker_id);
            nw_record_command_channel = replay_log;

            printf("\n//! starts to read recorded commands\n\n");
            while (1) {
                // Read logged commands by command_channel_recieve_command
                struct command_base *call_cmd = command_channel_receive_command(transfer_chan);
                if (call_cmd->api_id == COMMAND_HANDLER_API) {
                    if (call_cmd->command_id == COMMAND_END_MIGRATION) {
                        break;
                    }

                    /* replace explicit state */
                    handle_command(transfer_chan, replay_handle_pool,
                                   (struct command_channel *) nw_record_command_channel, call_cmd);
                } else {
                    command_channel_print_command(transfer_chan, call_cmd);
                    struct command_base *ret_cmd = command_channel_receive_command(transfer_chan);
                    command_channel_print_command(transfer_chan, ret_cmd);

                    // Replay the commands.
                    replay_command(transfer_chan, replay_handle_pool, (struct command_channel *) replay_log, call_cmd, ret_cmd);

                    command_channel_free_command(transfer_chan, ret_cmd);
                }
                command_channel_free_command(transfer_chan, call_cmd);
            }
            printf("\n//! finishes read of recorded commands\n\n");

            // TODO: For swapping we will need to selectively copy values back into the nw_global_handle_pool
            //  and then destroy the reply_handle_pool.
            nw_handle_pool_free(nw_global_handle_pool);
            nw_global_handle_pool = replay_handle_pool;

            {
                struct command_base *log_end = command_channel_new_command(chan, sizeof(struct command_base), 0);
                log_end->api_id = COMMAND_HANDLER_API;
                log_end->thread_id = cmd->thread_id;
                log_end->command_id = COMMAND_END_MIGRATION;
                /* notify guestlib of completion */
                command_channel_send_command(chan, log_end);
            }
            command_channel_free(transfer_chan);
            break;

        case COMMAND_END_MIGRATION:
            // TODO: Move this command into a handler guestlib/src/init.c
            nw_end_migration_flag = 1;
            break;

        case COMMAND_HANDLER_REPLACE_EXPLICIT_STATE:
            ava_handle_replace_explicit_state(chan, handle_pool, (struct ava_replay_command_t *) cmd);
            break;

        case COMMAND_START_LIVE_MIGRATION:
            transfer_chan = (struct command_channel *)command_channel_socket_tcp_migration_new(nw_worker_id, 1);
            struct timeval start, end;

            FILE *fd;
            fd = fopen("migration.log", "a");
            gettimeofday(&start, NULL);
            // Initiate the live migration
            {
                struct command_base *log_init = command_channel_new_command(transfer_chan, sizeof(struct command_base), 0);
                log_init->api_id = COMMAND_HANDLER_API;
                log_init->command_id = COMMAND_ACCEPT_LIVE_MIGRATION;
                // TODO: send more worker information
                command_channel_send_command(transfer_chan, log_init);
                DEBUG_PRINT("sent init migration message to target\n");
            }

            // Extract recorded commands and exlicit objects
            ava_extract_objects_in_pair(transfer_chan, nw_record_command_channel,
                    nw_handle_pool_get_live_handles(nw_global_handle_pool));
            DEBUG_PRINT("sent recorded commands to target\n");

            {
                struct command_base *log_end = command_channel_new_command(transfer_chan, sizeof(struct command_base), 0);
                log_end->api_id = COMMAND_HANDLER_API;
                log_end->command_id = COMMAND_END_LIVE_MIGRATION;
                command_channel_send_command(transfer_chan, log_end);
                DEBUG_PRINT("sent end migration message to target\n");
            }

            /* notify guestlib of completion */
            {
                struct command_base *log_end = command_channel_new_command(chan, sizeof(struct command_base), 0);
                log_end->api_id = COMMAND_HANDLER_API;
                log_end->thread_id = cmd->thread_id;
                log_end->command_id = COMMAND_END_MIGRATION;
                command_channel_send_command(chan, log_end);
                // TODO: guestlib reconnect to new worker
            }
            gettimeofday(&end, NULL);
            printf("migration takes %lf\n", ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0));
            fprintf(fd, "[%d] migration takes %lf\n", nw_worker_id, ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0));
            fclose(fd);

            break;

        case COMMAND_END_LIVE_MIGRATION:
            printf("\n//! finishes live migration\n\n");
            // TODO: target worker connects to guestlib
            usleep(1000000); // enough time for source worker to print out migration time
            exit(0);
            break;

        case COMMAND_ACCEPT_LIVE_MIGRATION:
            printf("\n//! starts to accept incoming commands\n\n");
            break;

        case COMMAND_HANDLER_RECORDED_PAIR:
            {
                struct ava_replay_command_pair_t *combine = (struct ava_replay_command_pair_t *)cmd;
                struct command_base *call_cmd = command_channel_get_buffer(chan, (struct command_base *)combine, combine->call_cmd);
                struct command_base *ret_cmd = command_channel_get_buffer(chan, (struct command_base *)combine, combine->ret_cmd);
                command_channel_print_command(chan, call_cmd);
                command_channel_print_command(chan, ret_cmd);

                printf("replay command <%ld, %lx>\n", call_cmd->command_id, call_cmd->region_size);

                // Replay the commands.
                replay_command(chan, nw_global_handle_pool, (struct command_channel *)nw_record_command_channel, call_cmd, ret_cmd);
            }
            break;

        default:
            DEBUG_PRINT("Unknown internal command: %lu", cmd->command_id);
            exit(0);
    }
}

EXPORTED_WEAKLY void init_internal_command_handler() {
    // TODO: currently guestlib initiates the migration. Should let hypervisor
    // or manager initiate it.
    register_command_handler(COMMAND_HANDLER_API, internal_api_handler, NULL, NULL);
}
