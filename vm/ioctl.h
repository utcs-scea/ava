#ifndef __VGPU_IOCTL_H__
#define __VGPU_IOCTL_H__

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#endif

#include <linux/kvm.h>

/* To guestdrv */

#define IOCTL_GET_VM_ID _IO(VGPU_DRIVER_MAJOR, 0x17)
#define IOCTL_REQUEST_SHM _IOR(VGPU_DRIVER_MAJOR, 0x16, size_t)
#define IOCTL_REQUEST_ZCOPY _IO(VGPU_DRIVER_MAJOR, 0x18)
#define IOCTL_GET_ZCOPY_PHY_ADDR _IOW(VGPU_ZCOPY_DRIVER_MAJOR, 0x10D, uintptr_t *)

/* To host KVM */

#define KVM_NOTIFY_NEW_WORKER _IOR(KVMIO, 0x107, int64_t)
#define PY_KVM_NOTIFY_NEW_WORKER 0x107

#define KVM_NOTIFY_VM_EXIT _IO(KVMIO, 0x106)
#define KVM_GET_VM_ID _IO(KVMIO, 0x10B)
#define KVM_SET_VM_GUEST_CID _IOR(KVMIO, 0x10C, int)

#define KVM_GET_ZCOPY_PHY_ADDR _IOW(VGPU_ZCOPY_DRIVER_MAJOR, 0x10D, uintptr_t *)

#define KVM_SET_SCHEDULING_POLICY _IOR(KVMIO, 0x10D, uintptr_t)
#define KVM_REMOVE_SCHEDULING_POLICY _IOR(KVMIO, 0x10F, int)
#define KVM_ATTACH_BPF _IOR(KVMIO, 0x10E, uintptr_t)
#define KVM_DETACH_BPF _IOR(KVMIO, 0x110, int)

/* APIs */

#define IOCTL_TF_PY_CMD 0x54
#define IOCTL_TF_PY_CALLBACK 0x55

#endif
