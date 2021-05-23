#ifndef __VGPU_DEVCONF_H__
#define __VGPU_DEVCONF_H__

/******************************************************************************
 * Measurements
 *****************************************************************************/

/**
 * Define to measure MMIO accesses to virtual transport device.
 * qemu/accel/kvm/kvm-all.c
 */
#undef QEMU_MMIO_COUNTER

/**
 * Define to measure in-kernel scheduling policy.
 * kvm/arch/x86/kvm/kvm_vgpu_measure.c
 */
#undef KVM_MEASURE_POLICY

/**
 * Interposition flags
 * We are refactoring the command scheduler and will transit to new
 * interposition method finally. The old scheduler does not have an
 * in-kernel buffer for forwarded commands, which does not meet the
 * normal design of schedulers.
 */
#undef AVA_VSOCK_INTERPOSITION

#if defined(AVA_VSOCK_INTERPOSITION)
#undef AVA_VSOCK_INTERPOSITION_NOBUF
#else
#define AVA_VSOCK_INTERPOSITION_NOBUF
#endif

#if defined(AVA_VSOCK_INTERPOSITION) || defined(AVA_VSOCK_INTERPOSITION_NOBUF)
#define AVA_ENABLE_KVM_MEDIATION
#else
#undef AVA_ENABLE_KVM_MEDIATION
#endif

/**
 * Worker report flags
 */
#define ENABLE_REPORT_BATCH 0

/**
 * VM configurations
 */
#define MAX_VM_NUM 4

#define VGPU_DEV_NAME "ava-vdev"

#define VGPU_DRIVER_MAJOR 1
#define VGPU_DRIVER_MINOR 0
#define VGPU_DRIVER_PATCHLEVEL 0
#define VGPU_ZCOPY_DRIVER_MAJOR 150
#define VGPU_ZCOPY_DRIVER_MINOR 0

/******************************************************************************
 * Virtual transport device BAR size
 *****************************************************************************/

#define VGPU_REG_SIZE 0x10000
#define VGPU_IO_SIZE 0x80
#define VGPU_VRAM_SIZE 128

/******************************************************************************
 * Shared memory channel
 *****************************************************************************/

#define AVA_APP_SHM_SIZE_DEFAULT MB(256)
#define AVA_GUEST_SHM_SIZE MB(512)
#define AVA_HOST_SHM_SIZE ((size_t)AVA_GUEST_SHM_SIZE * MAX_VM_NUM)

/* DMA_CMA region, the default size is 32 MB.
 * The size cannot be larger than that specified in the boot parameter cma=nnM */
#define VGPU_ZERO_COPY_SIZE MB(16)

/* Worker manager */
#define WORKER_MANAGER_SOCKET_PATH "/tmp/worker_manager"
#define DEST_SERVER_IP "10.0.0.2"

/* Hardware */
#define KB(x) (x << 10)
#define MB(x) ((KB(x)) << 10)
#define GB(x) ((MB(x)) << 10)

#define DEVICE_MEMORY_TOTAL_SIZE GB(2)

/* fair scheduling */
#define US_TO_US(x) ((long)x)
#define MS_TO_US(x) (US_TO_US(x) * 1000L)
#define SEC_TO_US(x) (MS_TO_US(x) * 1000L)

#define AVA_SEND_QUEUE_SIZE_DEFAULT 512
#define GPU_SCHEDULE_PERIOD 5          /* millisecond */
#define DEVICE_TIME_MEASURE_PERIOD 500 /* millisecond */
#define DEVICE_TIME_DELAY_ADD 1        /* millisecond */
#define DEVICE_TIME_DELAY_MUL_DEC 2
#define GPU_SCHEDULE_PERIOD_USEC 200  /* microsecond */
#define DEVICE_TIME_DELAY_ADD_USEC 50 /* microsecond */

/* swapping */
#define SWAP_SELECTION_DELAY 50 /* millisecond */

/* rate throttling */
#define COMMAND_RATE_LIMIT_BASE 100                                                          /* per second */
#define COMMAND_RATE_PERIOD_INIT 20                                                          /* millisecond */
#define COMMAND_RATE_BUDGET_BASE (COMMAND_RATE_LIMIT_BASE * COMMAND_RATE_PERIOD_INIT / 1000) /* per period */
#define COMMAND_RATE_MEASURE_PERIOD 1000                                                     /* millisecond */

/* shares */
static const int PREDEFINED_RATE_SHARES[MAX_VM_NUM + 1] = {0, 1, 1, 1, 1};
static const int PREDEFINED_PRIORITIES[MAX_VM_NUM + 1] = {0, 1, 1, 1, 1};
static const uint64_t DEV_MEM_PARTITIONS[MAX_VM_NUM + 1] = {0, GB(2UL), GB(2UL)};

/* auxiliary */
#define vgpu_max(a, b)      \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

#ifdef __KERNEL__
#include <linux/kernel.h>
#endif

#endif
