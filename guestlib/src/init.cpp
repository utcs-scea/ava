#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <grpc++/grpc++.h>

#include <iostream>
#include <string.h>
#include <vector>

#include "guestlib.h"
#include "guest_config.h"
#include "common/linkage.h"
#include "common/cmd_handler.h"
#include "common/shadow_thread_pool.h"
#include "common/endpoint_lib.h"
#include "common/cmd_channel.h"
#include "common/cmd_channel_impl.h"

struct command_channel *chan;

struct param_block_info nw_global_pb_info = {0, 0};
extern int nw_global_vm_id;

static struct command_channel* channel_create()
{
    return chan;
}

EXPORTED_WEAKLY void nw_init_guestlib(intptr_t api_id)
{
    std::ios_base::Init();

    guestconfig::config = guestconfig::readGuestConfig();
    if (guestconfig::config == nullptr)
      exit(EXIT_FAILURE);
#ifdef DEBUG
    guestconfig::config->print();
#endif

#ifdef AVA_PRINT_TIMESTAMP
    struct timeval ts;
    gettimeofday(&ts, NULL);
#endif

    /* Create connection to worker and start command handler thread */
    if (guestconfig::config->channel_ == "TCP") {
      std::vector<struct command_channel*> channels = command_channel_socket_tcp_guest_new();
      chan = channels[0];
    }
    else if (guestconfig::config->channel_ == "SHM") {
        chan = command_channel_shm_new();
    }
    else if (guestconfig::config->channel_ == "VSOCK") {
        chan = command_channel_socket_new();
    }
    else {
        std::cerr << "Unsupported channel specified in "
                  << guestconfig::kConfigFilePath
                  << ", expect channel = [\"TCP\" | \"SHM\" | \"VSOCK\"]" << std::endl;
        exit(0);
    }
    if (!chan) {
      std::cerr << "Failed to create command channel" << std::endl;
      exit(1);
    }
    init_command_handler(channel_create);
    init_internal_command_handler();

    /* Send initialize API command to the worker */
    struct command_handler_initialize_api_command* api_init_command =
        (struct command_handler_initialize_api_command*)command_channel_new_command(
            nw_global_command_channel, sizeof(struct command_handler_initialize_api_command), 0);
    api_init_command->base.api_id = COMMAND_HANDLER_API;
    api_init_command->base.command_id = COMMAND_HANDLER_INITIALIZE_API;
    api_init_command->base.vm_id = nw_global_vm_id;
    api_init_command->new_api_id = api_id;
    api_init_command->pb_info = nw_global_pb_info;
    command_channel_send_command(chan, (struct command_base*)api_init_command);

#ifdef AVA_PRINT_TIMESTAMP
    struct timeval ts_end;
    gettimeofday(&ts_end, NULL);
	printf("loading_time: %f\n", ((ts_end.tv_sec - ts.tv_sec) * 1000.0 + (float) (ts_end.tv_usec - ts.tv_usec) / 1000.0));
#endif
}

EXPORTED_WEAKLY void nw_destroy_guestlib(void)
{
    /* Send shutdown command to the worker */
    /*
    struct command_base* api_shutdown_command = command_channel_new_command(nw_global_command_channel, sizeof(struct command_base), 0);
    api_shutdown_command->api_id = COMMAND_HANDLER_API;
    api_shutdown_command->command_id = COMMAND_HANDLER_SHUTDOWN_API;
    api_shutdown_command->vm_id = nw_global_vm_id;
    command_channel_send_command(chan, api_shutdown_command);
    api_shutdown_command = command_channel_receive_command(chan);
    */

    // TODO: This is called by the guestlib so destructor for each API. This is safe, but will make the handler shutdown when the FIRST API unloads when having it shutdown with the last would be better.
    destroy_command_handler();
}

/**
 * Starts migration process for test.
 * */
EXPORTED_WEAKLY void start_migration(void)
{
    int manager_fd = -1;
    uintptr_t new_worker_id;

    if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "LOCAL")) {
        struct sockaddr_in serv_addr;
        manager_fd = socket(AF_INET, SOCK_STREAM, 0);

        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons( 4000 );
        inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
        connect(manager_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "SHM") || !strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
        /**
         * Get manager's host address from ENV('AVA_MANAGER_ADDR').
         * The address can either be a full IP:port or only the port (3333),
         * but the IP address is always ignored.
         */
        char *manager_full_address;
        int manager_port;
        manager_full_address = getenv("AVA_MANAGER_ADDR");
        assert(manager_full_address != NULL && "AVA_MANAGER_ADDR is not set");
        parseServerAddress(manager_full_address, NULL, NULL, &manager_port);
        assert(manager_port > 0 && "Invalid manager port");

        /* connect worker manager and send original worker id. */
        struct sockaddr_vm sa;
        manager_fd = init_vm_socket(&sa, VMADDR_CID_HOST, manager_port);
        conn_vm_socket(manager_fd, &sa);
    }

    struct command_base* msg = command_channel_new_command(chan, sizeof(struct command_base), 0);
    msg->command_type = COMMAND_START_MIGRATION;
    *((uintptr_t *)msg->reserved_area) = nw_worker_id;
    send_socket(manager_fd, msg, sizeof(struct command_base));

    recv_socket(manager_fd, msg, sizeof(struct command_base));
    new_worker_id = *((uintptr_t *)msg->reserved_area);
    printf("start to migrate from worker@%d to worker@%lu\n", nw_worker_id, new_worker_id);
    command_channel_free_command((struct command_channel *)chan, msg);
    close(manager_fd);

    // TODO: connect to new worker (replace chan->worker_fd)
}

// TODO: Should be removed once the guestlib can register it's own separate command handler in a single file.
// Flag used to communicate with the INTERNAL command handler. It is declared at common/cmd_handler.c:148
extern int nw_end_migration_flag;

/**
 * Starts migration process for test. The original and target workers
 * are the same one.
 */
EXPORTED_WEAKLY void start_self_migration(void)
{
    nw_end_migration_flag = 0;
    struct command_base* msg = command_channel_new_command(chan, sizeof(struct command_base), 0);
    msg->api_id = COMMAND_HANDLER_API;
    msg->command_id = COMMAND_START_MIGRATION;
    msg->thread_id = shadow_thread_id(nw_shadow_thread_pool);
    command_channel_send_command(chan, msg);

    /* wait until the migration finishes */
    shadow_thread_handle_command_until(nw_shadow_thread_pool, nw_end_migration_flag);
}

/**
 * Starts live migration process for test. The target worker is on a remote machine and connected via TCP.
 */
EXPORTED_WEAKLY void start_live_migration(void)
{
    nw_end_migration_flag = 0;
    struct command_base* msg = command_channel_new_command(chan, sizeof(struct command_base), 0);
    msg->api_id = COMMAND_HANDLER_API;
    msg->command_id = COMMAND_START_LIVE_MIGRATION;
    msg->thread_id = shadow_thread_id(nw_shadow_thread_pool);
    command_channel_send_command(chan, msg);

    /* wait until the migration finishes */
    shadow_thread_handle_command_until(nw_shadow_thread_pool, nw_end_migration_flag);

    // FIXME: let target have enough time to replay the log
    usleep(5000000);

    // TODO: reconnect to new worker
}
