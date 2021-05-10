#include "cudart_10.1_utilities.hpp"

#include <fatbinary.h>
#include <plog/Log.h>
#include <stdlib.h>

#include <stdexcept>

size_t __helper_fatbin_size(const void *cubin) {
  struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)cubin;
  return fbh->fatSize + fbh->headerSize;
}

void __helper_print_kernel_info(struct fatbin_function *func, void **args) {
  LOG_DEBUG << "function metadata (" << (void *)func << ") for local " << func->hostfunc << ", cufunc "
            << (void *)func->cufunc << ", argc " << func->argc;
  int i;
  for (i = 0; i < func->argc; i++) {
    LOG_DEBUG << "arg[" << i << "] is " << (func->args[i].is_handle ? "" : "not ")
              << "a handle, size = " << func->args[i].size << ", ptr = " << args[i]
              << ", content = " << *((void **)args[i]);
  }
}

cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret = (cudaError_t)CUDA_ERROR_PROFILER_ALREADY_STOPPED;

  if (func == NULL) return (cudaError_t)CUDA_ERROR_INVALID_PTX;

  if (func->hostfunc != hostFun) {
    LOG_ERROR << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc << " (device func "
              << (void *)func->cufunc << ")";
  } else {
    LOG_DEBUG << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc;
  }
  __helper_print_kernel_info(func, args);
  ret = (cudaError_t)cuLaunchKernel(func->cufunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                    sharedMem, (CUstream)stream, args, NULL);

  return ret;
}

int __helper_cubin_num(void **cubin_handle) {
  int num = 0;
  while (cubin_handle[num] != NULL) num++;
  return num;
}

void __helper_print_fatcubin_info(void *fatCubin, void **ret) {
  struct fatbin_wrapper *wp = (struct fatbin_wrapper *)fatCubin;
  printf("fatCubin_wrapper=%p, []={.magic=0x%X, .seq=%d, ptr=0x%lx, data_ptr=0x%lx}\n", fatCubin, wp->magic, wp->seq,
         wp->ptr, wp->data_ptr);
  struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)wp->ptr;
  printf("fatBinaryHeader={.magic=0x%X, version=%d, headerSize=0x%x, fatSize=0x%llx}\n", fbh->magic, fbh->version,
         fbh->headerSize, fbh->fatSize);
  char *fatBinaryEnd = (char *)(wp->ptr + fbh->headerSize + fbh->fatSize);
  printf("fatBin=0x%lx--0x%lx\n", wp->ptr, (int64_t)fatBinaryEnd);

  fatBinaryEnd = (char *)(wp->ptr);
  int i, j;
  for (i = 0; i < 100; i++)
    if (fatBinaryEnd[i] == 0x7F && fatBinaryEnd[i + 1] == 'E' && fatBinaryEnd[i + 2] == 'L') {
      printf("ELF header appears at 0x%d (%lx): \n", i, (uintptr_t)wp->ptr + i);
      break;
    }
  for (j = i; j < i + 32; j++) printf("%.2X ", fatBinaryEnd[j] & 0xFF);
  printf("\n");

  printf("ret=%p\n", ret);
  printf("fatCubin=%p, *ret=%p\n", (void *)fatCubin, *ret);
}

void __helper_unregister_fatbin(void **fatCubinHandle) {
  // free(fatCubinHandle);
  return;
}

void __helper_parse_function_args(const char *name, struct kernel_arg *args) {
  int i = 0, skip = 0;

  int argc = 0;
  if (strncmp(name, "_Z", 2)) abort();
  LOG_DEBUG << "Parse CUDA kernel " << name;

  i = 2;
  while (i < strlen(name) && isdigit(name[i])) {
    skip = skip * 10 + name[i] - '0';
    i++;
  }

  i += skip;
  while (i < strlen(name)) {
    switch (name[i]) {
    case 'P':
      args[argc++].is_handle = 1;

      /* skip qualifiers */
      if (strchr("rVK", name[i + 1]) != NULL) i++;

      if (i + 1 < strlen(name) && (strchr("fijl", name[i + 1]) != NULL))
        i++;
      else if (i + 1 < strlen(name) && isdigit(name[i + 1])) {
        skip = 0;
        while (i + 1 < strlen(name) && isdigit(name[i + 1])) {
          skip = skip * 10 + name[i + 1] - '0';
          i++;
        }
        i += skip;
      } else
        abort();
      break;

    case 'f': /* float */
    case 'i': /* int */
    case 'j': /* unsigned int */
    case 'l': /* long */
      args[argc++].is_handle = 0;
      break;

    case 'S':
      args[argc++].is_handle = 1;
      while (i < strlen(name) && name[i] != '_') i++;
      break;

    case 'v':
      i = strlen(name);
      break;

    case 'r': /* restrict (C99) */
    case 'V': /* volatile */
    case 'K': /* const */
      break;

    default:
      abort();
    }
    i++;
  }

  for (i = 0; i < argc; i++) {
    LOG_DEBUG << "function arg[" << i << "] is " << (args[i].is_handle == 1 ? "" : "not ") << "a handle";
  }
}

size_t __helper_launch_extra_size(void **extra) {
  size_t size = 1;
  while (extra[size - 1] != CU_LAUNCH_PARAM_END) size++;
  return size;
}

void *__helper_cu_mem_host_alloc_portable(size_t size) {
  void *p = aligned_alloc(64, size);
  assert(p);
  return p;
}

void __helper_cu_mem_host_free(void *ptr) { free(ptr); }

void __helper_assosiate_function(GHashTable *funcs, struct fatbin_function **func, void *local,
                                 const char *deviceName) {
  if (*func != NULL) {
    LOG_DEBUG << "Function (" << deviceName << ") metadata (" << local << ") already exists";
    return;
  }

  *func = (struct fatbin_function *)g_hash_table_lookup(funcs, deviceName);
  assert(*func && "device function not found!");
}

void __helper_register_function(struct fatbin_function *func, const char *hostFun, CUmodule module,
                                const char *deviceName) {
  // Empty fatbinary
  if (!module) {
    LOG_DEBUG << "Register a fat binary from a empty module";
    return;
  }

  if (func == NULL) {
    LOG_FATAL << "fatbin_function is NULL";
    throw std::invalid_argument("received empty fatbin_function");
  }

  // Only register the first host function
  if (func->hostfunc != NULL) return;

  CUresult ret = cuModuleGetFunction(&func->cufunc, module, deviceName);
  assert(ret == CUDA_SUCCESS);
  LOG_DEBUG << "register host func " << std::hex << (uintptr_t)hostFun << " -> device func " << (uintptr_t)func->cufunc;
  func->hostfunc = (void *)hostFun;
  func->module = module;
}

/**
 * Saves the async buffer information into the list inside the stream's
 * metadata.
 */
void __helper_register_async_buffer(struct async_buffer_list *buffers, void *buffer, size_t size) {
  assert(buffers->num_buffers < MAX_ASYNC_BUFFER_NUM);
  int idx = (buffers->num_buffers)++;
  LOG_VERBOSE << "Register async buffer [" << idx << "] address = " << buffer << ", size = " << size;
  buffers->buffers[idx] = buffer;
  buffers->buffer_sizes[idx] = size;
}

struct async_buffer_list *__helper_load_async_buffer_list(struct async_buffer_list *buffers) {
  if (buffers->num_buffers == 0) return NULL;

  LOG_DEBUG << "Load " << buffers->num_buffers << " async buffers";
  struct async_buffer_list *new_copy = (struct async_buffer_list *)malloc(sizeof(struct async_buffer_list));
  memcpy(new_copy, buffers, sizeof(struct async_buffer_list));
  memset(buffers, 0, sizeof(struct async_buffer_list));

  return new_copy;
}
