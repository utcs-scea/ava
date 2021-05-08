/*
 * desc: demo linux device driver for QEMU virtual GPU device.
 *
 */

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <asm/dma.h>
#include <asm/page.h>
#include <asm/uaccess.h>
#include <linux/dma-contiguous.h>
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/uaccess.h>

#include "common/devconf.h"
#include "common/ioctl.h"

#define DEVICE_NAME "ava_zcopy"
#define CLASS_NAME "ava_zcopy"

static struct class *dev_class;
static struct device *dev_node;

static void *zcopy_va;
static dma_addr_t dma_handle;

static int zcopy_open(struct inode *inode, struct file *filp) {
  pr_info("[ava-zcopy] open zcopy device always succeeds\n");
  return 0;
}

static int zcopy_release(struct inode *inode, struct file *filp) { return 0; }

static int zcopy_mmap(struct file *filp, struct vm_area_struct *vma) {
  if (VGPU_ZERO_COPY_SIZE != (vma->vm_end - vma->vm_start)) {
    pr_err("block size does not match\n");
    return -EAGAIN;
  }

  vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

  return dma_mmap_coherent(dev_node, vma, zcopy_va, dma_handle, VGPU_ZERO_COPY_SIZE);
}

static long zcopy_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
  int r = -EINVAL;
  uintptr_t phys;

  switch (cmd) {
  case KVM_GET_ZCOPY_PHY_ADDR:
    phys = virt_to_phys(zcopy_va);
    copy_to_user((void *)arg, (void *)&phys, sizeof(phys));
    r = 0;
    break;

  default:
    printk("unsupported IOCTL command\n");
  }

  return r;
}

static char *mod_dev_node(struct device *dev, umode_t *mode) {
  if (mode) *mode = 0666;
  return NULL;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = zcopy_open,
    .mmap = zcopy_mmap,
    .release = zcopy_release,
    .unlocked_ioctl = zcopy_ioctl,
};

static int __init zcopy_init(void) {
  int err = -ENOMEM;

  register_chrdev(VGPU_ZCOPY_DRIVER_MAJOR, DEVICE_NAME, &fops);
  pr_info("[ava-zcopy] registered zero copy device with major number %d\n", VGPU_ZCOPY_DRIVER_MAJOR);

  if (!(dev_class = class_create(THIS_MODULE, CLASS_NAME))) {
    pr_err("class_create error\n");
    goto unregister_dev;
  }
  dev_class->devnode = mod_dev_node;

  if (!(dev_node = device_create(dev_class, NULL, MKDEV(VGPU_ZCOPY_DRIVER_MAJOR, VGPU_ZCOPY_DRIVER_MINOR), NULL,
                                 DEVICE_NAME))) {
    pr_err("device_create error\n");
    goto destroy_class;
  }

  /* allocate physically contiguous memory region. */
  zcopy_va = dma_alloc_coherent(NULL, VGPU_ZERO_COPY_SIZE, &dma_handle, GFP_USER);
  if (!zcopy_va) {
    pr_err("dma_alloc error\n");
    goto destroy_device;
  } else {
    pr_info("[ava-zcopy] zero_copy region pa = 0x%lx, va = 0x%lx\n", (uintptr_t)virt_to_phys(zcopy_va),
            (uintptr_t)zcopy_va);
  }

  return 0;

destroy_device:
  device_destroy(dev_class, MKDEV(VGPU_ZCOPY_DRIVER_MAJOR, VGPU_ZCOPY_DRIVER_MINOR));

destroy_class:
  class_unregister(dev_class);
  class_destroy(dev_class);

unregister_dev:
  unregister_chrdev(VGPU_ZCOPY_DRIVER_MAJOR, DEVICE_NAME);
  return err;
}

static void __exit zcopy_fini(void) {
  dma_free_coherent(NULL, VGPU_ZERO_COPY_SIZE, zcopy_va, dma_handle);

  device_destroy(dev_class, MKDEV(VGPU_ZCOPY_DRIVER_MAJOR, VGPU_ZCOPY_DRIVER_MINOR));
  class_unregister(dev_class);
  class_destroy(dev_class);
  unregister_chrdev(VGPU_ZCOPY_DRIVER_MAJOR, DEVICE_NAME);
}

module_init(zcopy_init);
module_exit(zcopy_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Zero-copy module for QEMU virtual GPU device");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "." __stringify(0) "." __stringify(
    0) "."
       "0");
