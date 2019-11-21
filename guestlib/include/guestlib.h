#ifndef __VGPU_GUESTLIB_H__
#define __VGPU_GUESTLIB_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void nw_init_guestlib(intptr_t api_id);
void nw_destroy_guestlib(void);
void start_migration(void);
void start_self_migration(void);
void start_live_migration(void);

#ifdef __cplusplus
}
#endif

#endif
