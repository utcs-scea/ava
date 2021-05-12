#ifndef __VGPU_LDD_H__
#define __VGPU_LDD_H__

#include <asm/uaccess.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/sched.h>
#include <linux/semaphore.h>
#include <linux/uaccess.h>

#include "common/devconf.h"
#include "common/guest_mem.h"

struct mem_region {
  const char *name;
  void __iomem *base_addr;
  resource_size_t start;
  resource_size_t len;
  void __iomem *cur_addr;
  resource_size_t cur_offset;
  struct mutex lock;
};

enum alloc_type_t {
  ALLOC_TYPE_UNSPEC = 0,
  ALLOC_TYPE_SHM,
  ALLOC_TYPE_ZCOPY,
};

struct app_info {
  enum alloc_type_t alloc_type;
  /* current offset of free parameter block */
  uintptr_t free_pblock_offset;
  size_t pblock_size;
};

struct vgpu_dev {
  int vm_id;

  struct mem_region *reg;   /* BAR2 register region */
  struct mem_region *shm;   /* BAR4 register region */
  struct mem_region *zcopy; /* BAR5 zero-copy region */
};

#endif
