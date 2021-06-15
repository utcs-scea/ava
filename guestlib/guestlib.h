#ifndef __VGPU_GUESTLIB_H__
#define __VGPU_GUESTLIB_H__

#include <stdint.h>

#include "common/cmd_channel.hpp"
#include "migration.h"

#ifdef __cplusplus
#include <vector>

extern "C" {
#endif

void init_guestlib(void);
void destroy_guestlib(void);
void nw_init_log();
void nw_init_guestlib(intptr_t api_id);
void nw_destroy_guestlib(void);

#ifdef __cplusplus
}

std::vector<struct command_channel *> command_channel_socket_tcp_guest_new();
struct command_channel *command_channel_shm_guest_new();
#endif

#endif
