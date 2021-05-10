/*
 * desc: Linux device driver for QEMU virtual transport device.
 *
 */

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include "vgpu_ldd.h"

#include <asm/barrier.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include <linux/cdev.h>
#include <linux/circ_buf.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/module.h>
#include <linux/notifier.h>
#include <linux/pci.h>
#include <linux/reboot.h>
#include <linux/sched.h>
#include <linux/uaccess.h>

#include "access.h"
#include "common/devconf.h"
#include "common/ioctl.h"
#include "common/register.h"
#include "common/vmcall_ext.h"

#define VGPU_DEVS_NUM 1

static struct mem_region bar_reg;
static struct mem_region bar_shm;
static struct mem_region bar_zcopy;

static struct vgpu_dev vgpu_dev;

static struct pci_device_id vgpu_id_table[] = {
    {0x2222, 0x2223, PCI_ANY_ID, PCI_ANY_ID, 0, 0, 0},
    {0},
};
MODULE_DEVICE_TABLE(pci, vgpu_id_table);

int vgpu_bar_probe(struct pci_dev *pdev, struct mem_region *bar, unsigned bar_num) {
  struct device *dev = &pdev->dev;

  bar->start = pci_resource_start(pdev, bar_num);
  bar->len = pci_resource_len(pdev, bar_num);
  bar->base_addr = pci_iomap(pdev, bar_num, 0);
  if (!bar->base_addr) {
    dev_err(dev, "failed to iomap %s region of size 0x%lx\n", bar->name, (unsigned long)bar->len);
    return -EBUSY;
  }
  bar->cur_addr = bar->base_addr;
  bar->cur_offset = 0;
  mutex_init(&bar->lock);
  dev_info(dev, "%s bar base = 0x%lx\n", bar->name, (unsigned long)bar->base_addr);
  dev_info(dev, "%s bar start = 0x%lx, len = %lu\n", bar->name, (unsigned long)bar->start, (unsigned long)bar->len);

  return 0;
}

static int vgpu_probe(struct pci_dev *pdev, const struct pci_device_id *pdev_id) {
  int err;
  struct device *dev = &pdev->dev;

  if ((err = pci_enable_device(pdev))) {
    dev_err(dev, "pci_enable_device probe error %d for device %s\n", err, pci_name(pdev));
    return err;
  }

  if ((err = pci_request_regions(pdev, VGPU_DEV_NAME)) < 0) {
    dev_err(dev, "pci_request_regions err %d\n", err);
    goto pci_disable;
  }

  /* BAR2: registers */
  vgpu_dev.reg = &bar_reg;
  bar_reg.name = "reg";
  if (vgpu_bar_probe(pdev, &bar_reg, 2) < 0) goto pci_release;

  /* BAR4: data store ram */
  vgpu_dev.shm = &bar_shm;
  bar_shm.name = "shm";
  if (vgpu_bar_probe(pdev, &bar_shm, 4) < 0) goto pci_release;

  /* BAR5: zero-copy ram */
  if (DRM_READ8(vgpu_dev.reg, REG_ZERO_COPY)) {
    vgpu_dev.zcopy = &bar_zcopy;
    bar_zcopy.name = "zcopy";
    if (vgpu_bar_probe(pdev, &bar_zcopy, 5) < 0) goto pci_release;
  } else {
    vgpu_dev.zcopy = NULL;
  }

  return 0;

pci_release:
  pci_release_regions(pdev);
pci_disable:
  pci_disable_device(pdev);
  return -EBUSY;
}

static void vgpu_remove(struct pci_dev *pdev) {
  pci_iounmap(pdev, bar_reg.base_addr);
  pci_release_regions(pdev);
  pci_disable_device(pdev);
}

static struct pci_driver vgpu_pci_driver = {
    .name = VGPU_DEV_NAME,
    .probe = vgpu_probe,
    .id_table = vgpu_id_table,
    .remove = vgpu_remove,
};

static int vgpu_open(struct inode *inode, struct file *filp) {
  struct app_info *app_info;

  if (MINOR(inode->i_rdev) != VGPU_DRIVER_MINOR) {
    pr_info("minor: %d\n", VGPU_DRIVER_MINOR);
    return -ENODEV;
  }

  /* descriptor slab related */
  app_info = kzalloc(sizeof(struct app_info), GFP_KERNEL);
  filp->private_data = app_info;

  return 0;
}

static int vgpu_mmap(struct file *filp, struct vm_area_struct *vma) {
  struct app_info *app_info = filp->private_data;
  unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;

  AVA_DEBUG.printf("mmap type=%d\n", app_info->alloc_type);
  switch (app_info->alloc_type) {
  case ALLOC_TYPE_SHM:
    if (app_info->pblock_size != (vma->vm_end - vma->vm_start)) {
      printk("block size does not match, try again\n");
      return -EAGAIN;
    }
    offset += (unsigned long)(vgpu_dev.shm->start + app_info->free_pblock_offset);
    AVA_DEBUG.printf("mmap block offset=%lx\n", app_info->free_pblock_offset);

    app_info->alloc_type = ALLOC_TYPE_UNSPEC;
    break;

  case ALLOC_TYPE_ZCOPY:
    /* mmap zero-copy region */
    if (vgpu_dev.zcopy == NULL) {
      pr_err("zero-copy region is not plugged\n");
      return -ENODEV;
    }
    if (vgpu_dev.zcopy->len != (vma->vm_end - vma->vm_start)) {
      pr_err("block size does not match, try again\n");
      return -EAGAIN;
    }
    offset += (unsigned long)vgpu_dev.zcopy->start;
    AVA_DEBUG.printf("mmap zero-copy vpa=%lx, spa=%lx\n", (uintptr_t)vgpu_dev.zcopy->base_addr,
                     DRM_READ64(vgpu_dev.reg, REG_ZERO_COPY_PHYS));
    app_info->alloc_type = 0;
    break;

  default:
    printk("allocation type is unset\n");
    return -EINVAL;
  }

  vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
  if (io_remap_pfn_range(vma, vma->vm_start, offset >> PAGE_SHIFT, vma->vm_end - vma->vm_start, vma->vm_page_prot))
    return -EAGAIN;

  return 0;
}

static int vgpu_release(struct inode *inode, struct file *filp) {
  kfree(filp->private_data);
  return 0;
}

static DEFINE_SPINLOCK(shm_lock);
static long vgpu_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
  int r = -EINVAL;
  struct app_info *app_info = filp->private_data;
  uintptr_t zcopy_phys;

  switch (cmd) {
  case IOCTL_REQUEST_SHM:
    app_info->alloc_type = ALLOC_TYPE_SHM;

    spin_lock(&shm_lock);
    /* TODO: better parameter block management.
     * The guestdrv statically partitions the shared memory buffer
     * and mmaps the subregion into the application's address space.
     */
    if (vgpu_dev.shm->cur_offset + arg > AVA_GUEST_SHM_SIZE) {
      printk("no enough parameter block space, allocate from offset 0x0\n");
      app_info->free_pblock_offset = 0;
      app_info->pblock_size = arg;
      vgpu_dev.shm->cur_offset += arg;
      r = 0;
    } else {
      r = app_info->free_pblock_offset = vgpu_dev.shm->cur_offset;
      app_info->pblock_size = arg;
      vgpu_dev.shm->cur_offset += arg;
    }
    spin_unlock(&shm_lock);
    break;

  case IOCTL_GET_VM_ID:
    r = vgpu_dev.vm_id;
    break;

  case IOCTL_REQUEST_ZCOPY:
    app_info->alloc_type = ALLOC_TYPE_ZCOPY;
    r = 0;
    break;

  case IOCTL_GET_ZCOPY_PHY_ADDR:
    if (vgpu_dev.zcopy == NULL) {
      r = -ENOMEM;
    } else {
      zcopy_phys = DRM_READ64(vgpu_dev.reg, REG_ZERO_COPY_PHYS);
      copy_to_user((void *)arg, (void *)&zcopy_phys, sizeof(zcopy_phys));
      r = 0;
    }
    break;

  default:
    printk("unsupported IOCTL command\n");
  }

  return r;
}

static const struct file_operations vgpu_ops = {
    .owner = THIS_MODULE,
    .open = vgpu_open,
    .mmap = vgpu_mmap,
    .release = vgpu_release,
    .unlocked_ioctl = vgpu_ioctl,
};

static int vgpu_dev_uevent(struct device *dev, struct kobj_uevent_env *env) {
  add_uevent_var(env, "DEVMODE=%#o", 0666);
  return 0;
}

static int vgpu_reboot_callback(struct notifier_block *self, unsigned long val, void *data) {
  DRM_WRITE8(vgpu_dev.reg, REG_MOD_EXIT, 1);
  return 0;
}

static struct notifier_block vgpu_reboot_notifier = {
    .notifier_call = vgpu_reboot_callback,
};

static struct cdev cdev;         /* char device abstraction */
static struct class *vgpu_class; /* linux device model */
static int vgpu_drv_major;

static int __init vgpu_init(void) {
  int err = -ENOMEM;

  /* obtain major */
  dev_t mjr = MKDEV(VGPU_DRIVER_MAJOR, 0);
  if ((err = alloc_chrdev_region(&mjr, 0, VGPU_DEVS_NUM, VGPU_DEV_NAME)) < 0) {
    pr_err("alloc_chrdev_region error\n");
    return err;
  }
  vgpu_drv_major = MAJOR(mjr);

  /* init char dev */
  cdev_init(&cdev, &vgpu_ops);
  cdev.owner = THIS_MODULE;

  {
    dev_t devt;
    devt = MKDEV(vgpu_drv_major, VGPU_DRIVER_MINOR);
    if ((err = cdev_add(&cdev, devt, 1))) {
      pr_err("cdev_add error\n");
      goto unregister_dev;
    }
  }

  /* sysfs entry */
  if (!(vgpu_class = class_create(THIS_MODULE, VGPU_DEV_NAME))) {
    pr_err("class_create error\n");
    goto delete_dev;
  }
  vgpu_class->dev_uevent = vgpu_dev_uevent;

  /* create udev node */
  {
    dev_t devt;
    devt = MKDEV(vgpu_drv_major, VGPU_DRIVER_MINOR);
    if (!(device_create(vgpu_class, NULL, devt, NULL, VGPU_DEV_NAME "%d", VGPU_DRIVER_MINOR))) {
      pr_err("device_create error\n");
      goto destroy_class;
    }
  }

  /* register PCI device driver */
  if ((err = pci_register_driver(&vgpu_pci_driver)) < 0) {
    pr_err("pci_register_driver error\n");
    goto destroy_device;
  }

  /* register VM in executor */
  DRM_WRITE8(vgpu_dev.reg, REG_MOD_INIT, 1);
  vgpu_dev.vm_id = DRM_READ32(vgpu_dev.reg, REG_VM_ID);
  AVA_DEBUG.printf("assigned vm id=%d\n", vgpu_dev.vm_id);

  register_reboot_notifier(&vgpu_reboot_notifier);

  return 0;

destroy_device:
  device_destroy(vgpu_class, MKDEV(vgpu_drv_major, VGPU_DRIVER_MINOR));

destroy_class:
  class_destroy(vgpu_class);

delete_dev:
  cdev_del(&cdev);

unregister_dev:
  unregister_chrdev_region(MKDEV(vgpu_drv_major, VGPU_DRIVER_MINOR), VGPU_DEVS_NUM);

  return err;
}

static void __exit vgpu_fini(void) {
  /* unregister VM in executor */
  DRM_WRITE8(vgpu_dev.reg, REG_MOD_EXIT, 1);

  pci_unregister_driver(&vgpu_pci_driver);
  device_destroy(vgpu_class, MKDEV(vgpu_drv_major, VGPU_DRIVER_MINOR));
  class_destroy(vgpu_class);
  cdev_del(&cdev);
  unregister_chrdev_region(MKDEV(vgpu_drv_major, VGPU_DRIVER_MINOR), VGPU_DEVS_NUM);
}

module_init(vgpu_init);
module_exit(vgpu_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Demo module for QEMU virtual GPU device");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(VGPU_DRIVER_MAJOR) "."
               __stringify(VGPU_DRIVER_MINOR) "."
               __stringify(VGPU_DRIVER_PATCHLEVEL) "."
               "0");
