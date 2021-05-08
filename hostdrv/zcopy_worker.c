#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "include/common/devconf.h"
#include "include/common/ioctl.h"

int main() {
  int zcopy_fd;
  int r;
  uintptr_t zcopy_phys;
  void *zcopy_base;

  zcopy_fd = open("/dev/ava_zcopy", O_RDWR);
  if (zcopy_fd < 0) {
    fprintf(stdout, "Zero-copy driver is not installed: %s\n", strerror(errno));
    return 0;
  }

  r = ioctl(zcopy_fd, KVM_GET_ZCOPY_PHY_ADDR, &zcopy_phys);
  if (r < 0) {
    fprintf(stderr, "ava_map_zero_copy_region failed\n");
  }
  zcopy_base = mmap(NULL, VGPU_ZERO_COPY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, zcopy_fd, 0);
  if (zcopy_base == MAP_FAILED) {
    fprintf(stderr, "ava mmap_zero_copy_region failed\n");
  } else {
    fprintf(stdout, "mmap zero copy region to 0x%lx, phys=0x%lx\n", (uintptr_t)zcopy_base, zcopy_phys);
  }

  fprintf(stdout, "Old value %d\n", *(int *)zcopy_base);
  *(int *)zcopy_base = 42;
  fprintf(stdout, "Stored value %d\n", *(int *)zcopy_base);

  /* unmap zero copy region */
  if (zcopy_fd > 0) munmap(zcopy_base, VGPU_ZERO_COPY_SIZE);

  close(zcopy_fd);

  return 0;
}
