#include "migration.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/common_context.h"
#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/shadow_thread_pool.hpp"
#include "guestlib/guest_context.h"

/**
 * Starts migration process for test.
 * */
EXPORTED_WEAKLY void start_migration(struct command_channel *chan) {
  int manager_fd = -1;
  uintptr_t new_worker_id;

  if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "LOCAL")) {
    struct sockaddr_in serv_addr;
    manager_fd = socket(AF_INET, SOCK_STREAM, 0);

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(4000);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
    connect(manager_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
  } else if (!strcmp(getenv("AVA_CHANNEL"), "SHM") || !strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
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

  struct command_base *msg = command_channel_new_command(chan, sizeof(struct command_base), 0);
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

// TODO: Should be removed once the guestlib can register it's own separate
// command handler in a single file. Flag used to communicate with the INTERNAL
// command handler. It is declared at common/cmd_handler.c:148
extern int nw_end_migration_flag;

/**
 * Starts migration process for test. The original and target workers
 * are the same one.
 */
EXPORTED_WEAKLY void start_self_migration(struct command_channel *chan) {
  auto common_context = ava::CommonContext::instance();
  nw_end_migration_flag = 0;
  struct command_base *msg = command_channel_new_command(chan, sizeof(struct command_base), 0);
  msg->api_id = COMMAND_HANDLER_API;
  msg->command_id = COMMAND_START_MIGRATION;
  msg->thread_id = shadow_thread_id(common_context->nw_shadow_thread_pool);
  command_channel_send_command(chan, msg);

  /* wait until the migration finishes */
  shadow_thread_handle_command_until(common_context->nw_shadow_thread_pool, nw_end_migration_flag);
}

// TODO(migration): instead of letting guestlib initiates the migration, we
// should have AvA manager start the process:
// 1. The AvA manager starts a target API server.
// 2. The AvA manager tells the source API server to start live migration with
// the target
//    API server's address provided.
// 3. The source API server suspends the API execution for the guestlib.
// 4. The source API server sends the API log to the target server to replay and
// asks the
//    guestlib to connect the target API server (it also closes the connection
//    to the guestlib).
// 5. The target API server replays the log and accepts the guestlib.
// 6. The target API server tells the guestlib to continue the API remoting.
//
// The close of connection between the source API server and guestlib in (4) is
// a bit tricy at this moment, as any channel connecton means a fault at an end
// and shuts down the other. The best approach is to close the channel by the
// guestlib, so that the source API server will shut down by itself.
EXPORTED_WEAKLY void start_live_migration(struct command_channel *chan) {
  auto common_context = ava::CommonContext::instance();
  nw_end_migration_flag = 0;
  struct command_base *msg = command_channel_new_command(chan, sizeof(struct command_base), 0);
  msg->api_id = COMMAND_HANDLER_API;
  msg->command_id = COMMAND_START_LIVE_MIGRATION;
  msg->thread_id = shadow_thread_id(common_context->nw_shadow_thread_pool);
  command_channel_send_command(chan, msg);

  /* wait until the migration finishes */
  shadow_thread_handle_command_until(common_context->nw_shadow_thread_pool, nw_end_migration_flag);

  // TODO(migration): reconnect to new worker to execute the left APIs.
  // Currently, the APIs after the migration point are still executed by
  // the original API server.

  // TODO(migration): the target API server notifies guestlib to continue.
  // FIXME: let target have enough time to replay the log
  usleep(5000000);
}
