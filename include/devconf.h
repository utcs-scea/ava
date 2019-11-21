#ifndef __VGPU_DEVCONF_H__
#define __VGPU_DEVCONF_H__

#include "ctype_util.h"
#include "debug.h"

/******************************************************************************
 * Measurements
 *****************************************************************************/

/**
 * Define to measure MMIO accesses to virtual transport device.
 * qemu/accel/kvm/kvm-all.c
 */
#undef QEMU_MMIO_COUNTER

/**
 * Interposition flags
 */
#define AVA_VSOCK_INTERPOSITION
#define ENABLE_KVM_MEDIATION 1
#define ENABLE_RATE_LIMIT    1
#define ENABLE_SWAP          1
#define ENABLE_REPORT_BATCH  0

#define MAX_VM_NUM           4

#define VGPU_DEV_NAME "ava-vdev"

#define VGPU_DRIVER_MAJOR 1
#define VGPU_DRIVER_MINOR 0
#define VGPU_DRIVER_PATCHLEVEL 0
#define VGPU_ZCOPY_DRIVER_MAJOR 150
#define VGPU_ZCOPY_DRIVER_MINOR 0

/******************************************************************************
 * Virtual transport device BAR size
 *****************************************************************************/

#define VGPU_REG_SIZE    0x10000
#define VGPU_IO_SIZE     0x80
#define VGPU_VRAM_SIZE   128

/******************************************************************************
 * Shared memory channel
 *****************************************************************************/

#define AVA_APP_SHM_SIZE_DEFAULT  MB(256)
#define AVA_GUEST_SHM_SIZE        MB(512)
#define AVA_HOST_SHM_SIZE         ((size_t)AVA_GUEST_SHM_SIZE * MAX_VM_NUM)

/* DMA_CMA region, the default size is 32 MB.
 * The size cannot be larger than that specified in the boot parameter cma=nnM */
#define VGPU_ZERO_COPY_SIZE       MB(16)

/* Worker manager */
#define WORKER_POOL_SIZE    1
#define WORKER_MANAGER_PORT 3333
#define WORKER_PORT_BASE    4000
#define WORKER_MANAGER_SOCKET_PATH "/tmp/worker_manager"
#define DEST_SERVER_IP      "10.0.0.2"

/* Hardware */
#define DEVICE_MEMORY_TOTAL_SIZE GB(2)

/* fair scheduling */
#define AVA_SEND_QUEUE_SIZE_DEFAULT 512
#define GPU_SCHEDULE_PERIOD         5     /* millisecond */
#define DEVICE_TIME_MEASURE_PERIOD  500   /* millisecond */
#define DEVICE_TIME_DELAY_ADD       1     /* millisecond */
#define DEVICE_TIME_DELAY_MUL_DEC   2
#define GPU_SCHEDULE_PERIOD_USEC    200   /* microsecond */
#define DEVICE_TIME_DELAY_ADD_USEC  50    /* microsecond */

/* swapping */
#define SWAP_SELECTION_DELAY        50    /* millisecond */

/* rate throttling */
#define COMMAND_RATE_LIMIT_BASE     100   /* per second */
#define COMMAND_RATE_PERIOD_INIT    20    /* millisecond */
#define COMMAND_RATE_BUDGET_BASE    (COMMAND_RATE_LIMIT_BASE * COMMAND_RATE_PERIOD_INIT / 1000)  /* per period */
#define COMMAND_RATE_MEASURE_PERIOD 1000  /* millisecond */

/* shares */
static const int PREDEFINED_RATE_SHARES[MAX_VM_NUM+1]  = {0, 1, 1, 1, 1};
static const int PREDEFINED_PRIORITIES[MAX_VM_NUM+1]   = {0, 1, 1, 1, 1};
static const uint64_t DEV_MEM_PARTITIONS[MAX_VM_NUM+1] = {0, GB(2UL), GB(2UL)};

/* auxiliary */
#define vgpu_max(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a > _b ? _a : _b; })

#ifdef __KERNEL__
    #include <linux/kernel.h>
#else
    #define max(a, b) vgpu_max(a, b)
#endif

#endif
