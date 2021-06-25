#ifndef _AVA_GUESTLIB_EXTENSIONS_GPU_ADDRESS_TRACKING_H_
#define _AVA_GUESTLIB_EXTENSIONS_GPU_ADDRESS_TRACKING_H_
#include <glib.h>

#include <cstdint>

void __helper_save_gpu_address_range(uint64_t dptr, size_t bytesize, void *ret);
bool is_gpu_address(uint64_t ptr);
void __helper_remove_gpu_address_range(uint64_t dptr);
void gpu_address_tracking_init();
void gpu_address_tracking_fini();

#endif  // _AVA_GUESTLIB_EXTENSIONS_GPU_ADDRESS_TRACKING_H_
