#ifndef AVA_GUESTLIB_MIGRATION_H_
#define AVA_GUESTLIB_MIGRATION_H_

#include "common/cmd_channel.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void start_migration(struct command_channel *chan);
void start_self_migration(struct command_channel *chan);

/**
 * start_live_migration - Starts live migration process for test
 *
 * @chan: command channel to the original API server
 *
 * This sends an internal command (`COMMAND_START_LIVE_MIGRATION`) to the original
 * API server. The source API server establishes a socket_tcp_migration channel
 * to the target worker and transfers the command replay log.
 */
void start_live_migration(struct command_channel *chan);

#ifdef __cplusplus
}
#endif

#endif  // AVA_GUESTLIB_MIGRATION_H_
