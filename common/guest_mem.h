#ifndef __VGPU_GUEST_MEM_H__
#define __VGPU_GUEST_MEM_H__

#ifdef __KERNEL__

#include <linux/list.h>

#else

#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>

#include "import/list.h"

#endif

#ifdef __cplusplus
extern "C" {
#endif

/* parameter data block */
struct param_block {
  struct list_head list;

  void *base;
  size_t size;
  uintptr_t offset; /* offset to vgpu_dev.dstore->base_addr */

  // management
  uintptr_t cur_offset; /* start from 0 */
};

struct block_seeker {
  uintptr_t global_offset; /* local_offset + param_block.offset */
  uintptr_t local_offset;  /* start from 0 */
  uintptr_t cur_offset;    /* start from local_offset + 0x4 */
};

struct param_block_info {
  uintptr_t param_local_offset;
  uintptr_t param_block_size;
};

struct sched_policy {
  char *module_name;
  uint64_t module_name_len;
  char *cb_struct; /* callback struct name */
  uint64_t cb_struct_len;
  char *consume_func_name;
  uint64_t consume_func_name_len;
};
#ifdef __cplusplus
}
#endif

#endif
