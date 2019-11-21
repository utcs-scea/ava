#ifndef __EXECUTOR_SHADOW_H__
#define __EXECUTOR_SHADOW_H__

#include <stdint.h>
#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vm_state {
    GHashTable *map_hash;
    uintptr_t map_addr;
};

#define HASH_TIMES 3
uintptr_t feistel_cipher(uintptr_t);

void addr_map_init(size_t);
uintptr_t addr_map(size_t, uintptr_t);
uintptr_t addr_demap(size_t, uintptr_t);
void addr_unmap(size_t, uintptr_t);

#ifdef __cplusplus
}
#endif

#endif
