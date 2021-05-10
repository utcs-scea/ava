#ifndef AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_
#define AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glib.h>

#define MAX_KERNEL_ARG 25
#define MAX_KERNEL_NAME_LEN 1024
#define MAX_ASYNC_BUFFER_NUM 16

struct fatbin_wrapper {
  uint32_t magic;
  uint32_t seq;
  uint64_t ptr;
  uint64_t data_ptr;
};

struct kernel_arg {
  char is_handle;
  uint32_t size;
};

struct fatbin_function {
  int argc;
  struct kernel_arg args[MAX_KERNEL_ARG];

  CUfunction cufunc;
  void *hostfunc;
  CUmodule module;
};

size_t __helper_fatbin_size(const void *cubin);

void __helper_print_kernel_info(struct fatbin_function *func, void **args);

cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream);

int __helper_cubin_num(void **cubin_handle);

void __helper_print_fatcubin_info(void *fatCubin, void **ret);

void __helper_unregister_fatbin(void **fatCubinHandle);

void __helper_parse_function_args(const char *name, struct kernel_arg *args);

size_t __helper_launch_extra_size(void **extra);

void *__helper_cu_mem_host_alloc_portable(size_t size);

void __helper_cu_mem_host_free(void *ptr);

void __helper_assosiate_function(GHashTable *funcs, struct fatbin_function **func, void *local, const char *deviceName);

void __helper_register_function(struct fatbin_function *func, const char *hostFun, CUmodule module,
                                const char *deviceName);

/* Async buffer address list */
struct async_buffer_list {
  int num_buffers;
  void *buffers[MAX_ASYNC_BUFFER_NUM]; /* array of buffer addresses */
  size_t buffer_sizes[MAX_ASYNC_BUFFER_NUM];
};

void __helper_register_async_buffer(struct async_buffer_list *buffers, void *buffer, size_t size);

struct async_buffer_list *__helper_load_async_buffer_list(struct async_buffer_list *buffers);

#endif  // AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_
