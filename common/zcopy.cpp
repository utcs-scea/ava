//
// Created by amp on 8/7/19.
//

#include "common/zcopy.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/devconf.h"
#include "common/ioctl.h"
#include "common/logging.h"

struct ava_zcopy_region {
  int fd;
  uintptr_t physical_base;
  void *base;
  size_t size;

  void *allocation_ptr;

  // TODO: The mutex will not work across the guest/worker boundary. We will
  // need to use an in region lock-free data structure. For the moment we are
  // assuming all alloc/free is done in the guest.
  // TODO: alloc and free should be lock-free.
  pthread_mutex_t lock;
};

#define ENCODED_PTR_OFFSET 4096  // One page (not actually important as long as it is > 0)

struct ava_zcopy_region *ava_zcopy_region_new_worker() {
  struct ava_zcopy_region *ret = malloc(sizeof(struct ava_zcopy_region));
  bzero(ret, sizeof(struct ava_zcopy_region));

  int r;
  pthread_mutex_init(&ret->lock, NULL);
  pthread_mutex_lock(&ret->lock);

  ret->fd = open("/dev/ava_zcopy", O_RDWR);
  if (ret->fd < 0) {
    AVA_ERROR << "Zero-copy driver is not installed: " << strerror(errno);
    return NULL;
  }

  r = ioctl(ret->fd, KVM_GET_ZCOPY_PHY_ADDR, &ret->physical_base);
  if (r < 0) {
    AVA_ERROR << "ava_map_zero_copy_region failed";
    return NULL;
  }

  ret->size = VGPU_ZERO_COPY_SIZE;
  ret->base = mmap(NULL, VGPU_ZERO_COPY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, ret->fd, 0);
  if (ret->base == MAP_FAILED) {
    AVA_ERROR << "ava mmap VGPU_ZERO_COPY_SIZE failed";
    return NULL;
  }

  ret->allocation_ptr = ret->base;
  pthread_mutex_unlock(&ret->lock);
  return ret;
}

struct ava_zcopy_region *ava_zcopy_region_new_guest() {
  struct ava_zcopy_region *ret = malloc(sizeof(struct ava_zcopy_region));
  bzero(ret, sizeof(struct ava_zcopy_region));

  int r;
  pthread_mutex_init(&ret->lock, NULL);
  pthread_mutex_lock(&ret->lock);

  ret->fd = open("/dev/scea-vgpu0", O_RDWR);
  if (ret->fd < 0) {
    AVA_ERROR << "VGPU driver is not installed, trying host driver.";
    free(ret);
    return ava_zcopy_region_new_worker();
  }

  ret->size = VGPU_ZERO_COPY_SIZE;
  r = ioctl(ret->fd, IOCTL_GET_ZCOPY_PHY_ADDR, &ret->physical_base);
  if (r < 0) {
    AVA_ERROR << "IOCTL_GET_ZCOPY_PHY_ADDR failed";
    return NULL;
  }

  r = ioctl(ret->fd, IOCTL_REQUEST_ZCOPY);
  ret->base = mmap(NULL, VGPU_ZERO_COPY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, ret->fd, 0);
  if (r < 0 || ret->base == MAP_FAILED) {
    AVA_ERROR << "ava mmap VGPU_ZERO_COPY_SIZE failed";
    return NULL;
  }

  ret->allocation_ptr = ret->base;
  pthread_mutex_unlock(&ret->lock);
  return ret;
}

void ava_zcopy_region_free_region(struct ava_zcopy_region *region) {
  assert(region != NULL);
  pthread_mutex_lock(&region->lock);
  if (region->base > 0) munmap(region->base, VGPU_ZERO_COPY_SIZE);

  close(region->fd);
  region->base = NULL;
  region->fd = -1;
  pthread_mutex_unlock(&region->lock);
  pthread_mutex_destroy(&region->lock);
  free(region);
}

void *ava_zcopy_region_alloc(struct ava_zcopy_region *region, size_t size) {
  assert(region != NULL && "The appropriate zero-copy driver may not be installed.");
  pthread_mutex_lock(&region->lock);
  // TODO: Align the allocated memory.
  void *ret = region->allocation_ptr;
  // TODO: Add header to each allocation so they can be freed.
  if (region->base != NULL && ret + size < region->base + region->size) {
    region->allocation_ptr += size;
  } else {
    ret = NULL;
    errno = ENOMEM;
  }
  pthread_mutex_unlock(&region->lock);
  return ret;
}

void ava_zcopy_region_free(struct ava_zcopy_region *region, void *ptr) {
  assert(region != NULL && "The appropriate zero-copy driver may not be installed.");
  pthread_mutex_lock(&region->lock);
  // TODO: Implement deallocation. This will be complicated by the need for
  // potential coordination between the guest and worker allocating in the same
  // region.
  pthread_mutex_unlock(&region->lock);
}

uintptr_t ava_zcopy_region_get_physical_address(struct ava_zcopy_region *region, const void *ptr) {
  assert(region != NULL && "The appropriate zero-copy driver may not be installed.");
  if (region->base == NULL || (ptr < region->base || ptr > region->base + region->size)) {
    errno = EFAULT;
    return 0;
  }
  return (uintptr_t)ptr - (uintptr_t)region->base + (uintptr_t)region->physical_base;
}

void *ava_zcopy_region_encode_position_independent(struct ava_zcopy_region *region, const void *ptr) {
  assert(region != NULL && "The appropriate zero-copy driver may not be installed.");
  if (ptr == NULL) return NULL;
  if (region->base == NULL || (ptr < region->base || ptr > region->base + region->size)) {
    errno = EFAULT;
    return 0;
  }
  // Add offset to avoid valid ptrs being encoded to NULL
  return (void *)((uintptr_t)ptr - (uintptr_t)region->base) + ENCODED_PTR_OFFSET;
}
void *ava_zcopy_region_decode_position_independent(struct ava_zcopy_region *region, const void *ptr) {
  assert(region != NULL && "The appropriate zero-copy driver may not be installed.");
  if (ptr == NULL) return NULL;
  if (region->base == NULL ||
      ((uintptr_t)ptr < ENCODED_PTR_OFFSET || (uintptr_t)ptr > ENCODED_PTR_OFFSET + region->size)) {
    errno = EFAULT;
    return 0;
  }
  // Remove offset from encoding
  return region->base + (uintptr_t)ptr - ENCODED_PTR_OFFSET;
}
