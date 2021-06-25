#include <cuda.h>
#include <glib.h>

#include "common/logging.h"

// address range
// assume there's no overlap in the address ranges
struct gpu_address_range {
  uintptr_t start;
  uintptr_t end;
};

GTree *gpu_address_set = nullptr;

static gint gpu_address_range_cmp(gconstpointer r1, gconstpointer r2, gpointer user_data) {
  long diff = ((uintptr_t)r1 - (uintptr_t)r2);
  if (diff < 0) return -1;
  if (diff > 0) return 1;
  return 0;
}

void gpu_address_tracking_init() {
  /* Save allocated GPU memory addresses */
  gpu_address_set = g_tree_new_full(gpu_address_range_cmp, NULL, NULL, g_free);
}

void gpu_address_tracking_fini() { g_tree_destroy(gpu_address_set); }

void __helper_save_gpu_address_range(uintptr_t dptr, size_t bytesize, void *ret) {
  CUresult *cu_ret = static_cast<CUresult *>(ret);
  if (cu_ret != nullptr && *cu_ret == CUDA_SUCCESS) {
    struct gpu_address_range *range = (struct gpu_address_range *)g_malloc(sizeof(struct gpu_address_range));
    range->start = dptr;
    range->end = dptr + bytesize;
    g_tree_insert(gpu_address_set, (gpointer)range->start, (gpointer)range);
    ava_debug("Save GPU address range [%lx, %lx)", range->start, range->end);
  }
}

static gint gpu_address_search_func(gconstpointer a, gconstpointer b) {
  struct gpu_address_range *r = (struct gpu_address_range *)g_tree_lookup(gpu_address_set, a);
  if (r->start > (uintptr_t)b) return -1;
  if (r->end <= (uintptr_t)b) return 1;
  return 0;
}

bool is_gpu_address(uintptr_t ptr) {
  gpointer res = g_tree_search(gpu_address_set, gpu_address_search_func, (gconstpointer)ptr);
  return res != nullptr;
}

void __helper_remove_gpu_address_range(uintptr_t dptr) {
  auto ret = g_tree_remove(gpu_address_set, (gpointer)dptr);
  if (!ret) {
    AVA_LOG_F(ERROR, "address {} is not tracked", dptr);
  }
}
