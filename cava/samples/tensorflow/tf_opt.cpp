// clang-format off
ava_name("CUDA Runtime for TensorFlow");
ava_version("10.1.0");
ava_identifier(TF_OPT);
ava_number(9);
ava_cxxflags(-I/usr/local/cuda-10.1/include -I${CMAKE_SOURCE_DIR}/cava/headers -DAVA_PRELOAD_CUBIN);
ava_libs(-L/usr/local/cuda-10.1/lib64 -lcudart -lcuda -lcublas -lcudnn -lcufft -lcurand -lcusparse -lcusolver);
ava_guestlib_srcs(extensions/cudnn_optimization.cpp extensions/tf_optimization.cpp extensions/cmd_batching.cpp);
ava_worker_srcs(extensions/cudnn_optimization.cpp extensions/tf_optimization.cpp extensions/cmd_batching.cpp);
ava_common_utility_srcs(extensions/cudart_10.1_utilities.cpp);
ava_export_qualifier();
ava_soname(libcuda.so libcuda.so.1 libcudart.so.10 libcudart.so.10.1 libcublas.so.10 libcublasLt.so.10 libcudnn.so.7 libcufft.so.10 libcurand.so.10 libcusolver.so.10 libcusparse.so.10);
// clang-format on

/**
 * This spec reads the dumped fat binaries and CUDA functions to
 * suppress the forwarding of __cudaRegister* APIs.
 * Compile by
 * ./nwcc samples/tensorflow/tf_opt.cpp -I /usr/local/cuda-10.0/include -I headers `pkg-config --cflags glib-2.0`
 *
 * Dependencies:
 * CUDA 10.1, cuDNN 7.6.5
 */

ava_non_transferable_types { ava_handle; }

size_t __args_index_0;
size_t __kernelParams_index_0;

ava_begin_utility;
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <glib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fatbinary.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cudnn.h>
#include <curand.h>
#include <cufft.h>
#include <cusparse.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "cudart_nw_internal.h"
#include "common/linkage.h"
#include "common/logging.h"
#include "common/extensions/tf_optimization.h"
#include "common/extensions/cmd_batching.h"
#include "common/extensions/cudart_10.1_utilities.hpp"

#if !defined(__dv)
#define __dv(v)
#endif /* !__dv */

// TODO(yuhc): Correctly generate code for union in struct (cudnnAlgorithm_t).
typedef union Algorithm {
  cudnnConvolutionFwdAlgo_t convFwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
  cudnnRNNAlgo_t RNNAlgo;
  cudnnCTCLossAlgo_t CTCLossAlgo;
};

extern GPtrArray *fatbin_handle_list;

struct call_configuration {
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem;
  void *stream;
};

extern GQueue *call_configuration_stack;
extern GQueue *convolution_descriptor_pool;
extern GQueue *idle_convolution_descriptor_pool;
extern GQueue *pooling_descriptor_pool;
extern GQueue *idle_pooling_descriptor_pool;
extern GQueue *tensor_descriptor_pool;
extern GQueue *idle_tensor_descriptor_pool;
extern GQueue *filter_descriptor_pool;
extern GQueue *idle_filter_descriptor_pool;
extern GQueue *cu_event_pool;
extern GQueue *idle_cu_event_pool;

extern cudaError_t cuda_last_error;

struct gpu_address_range {
  uintptr_t start;
  uintptr_t end;
};

extern GTree *gpu_address_set;
ava_end_utility;

ava_type(cudaError_t) { ava_success(cudaSuccess); }

ava_type(cublasStatus_t) { ava_success(CUBLAS_STATUS_SUCCESS); }

ava_type(cudnnStatus_t) { ava_success(CUDNN_STATUS_SUCCESS); }

ava_type(CUresult) { ava_success(CUDA_SUCCESS); }

ava_type(unsigned) { ava_success(CUDA_SUCCESS); }

ava_type(curandStatus_t) { ava_success(CURAND_STATUS_SUCCESS); }

ava_type(cufftResult) { ava_success(CUFFT_SUCCESS); }

ava_type(cusparseStatus_t) { ava_success(CUSPARSE_STATUS_SUCCESS); }

ava_type(cusolverStatus_t) { ava_success(CUSOLVER_STATUS_SUCCESS); }

typedef struct {
  /* read dumps */
  int num_fatbins;
  void *func_id;
  int fatfunction_fd;
  GHashTable *ht_name2idx;
  int fatbin_num_cur;

  /* argument types */
  GPtrArray *fatbin_funcs; /* for NULL, the hash table */
  int num_funcs;
  struct fatbin_function *func; /* for functions */

  /* global states */
  int cuinit_called;

  /* memory flags */
  int is_pinned;

  /* async buffers */
  struct async_buffer_list async_buffers;
} Metadata;

ava_register_metadata(Metadata);

ava_type(struct fatbin_wrapper) {
  struct fatbin_wrapper *ava_self;

  ava_field(magic);
  ava_field(seq);
  ava_field(ptr) {
    /* worker loads the fat binary from dump file */
    ava_self->ptr = 0;
  }
  ava_field(data_ptr) { ava_self->data_ptr = 0; }
}

ava_type(struct async_buffer_list) {
  struct async_buffer_list *ava_self;

  ava_field(num_buffers);
  ava_field(buffers) {
#warning Fix annotating an array of pointer
    ava_in;
    ava_buffer(ava_self->num_buffers);
    ava_element {
      ava_out;
      ava_buffer(ava_self->buffer_sizes[ava_index]);
      ava_lifetime_manual;
    }
  }
}

ava_type(struct cudaDeviceProp);

ava_type(struct cudaPointerAttributes) {
  ava_field(devicePointer) ava_opaque;
  ava_field(hostPointer) ava_opaque;
};

/* APIs for batching and pooling */

cudnnStatus_t __pool_cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count) {
  ava_argument(convDesc) {
    ava_out;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count) {
  ava_async;
  ava_argument(convDesc) {
    ava_in;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count) {
  ava_argument(filterDesc) {
    ava_out;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count) {
  ava_async;
  ava_argument(filterDesc) {
    ava_in;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count) {
  ava_argument(poolingDesc) {
    ava_out;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count) {
  ava_async;
  ava_argument(poolingDesc) {
    ava_in;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count) {
  ava_argument(tensorDesc) {
    ava_out;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

cudnnStatus_t __pool_cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count) {
  ava_async;
  ava_argument(tensorDesc) {
    ava_in;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

CUresult __pool_cuEventCreate(CUevent *phEvent, size_t count) {
  ava_argument(phEvent) {
    ava_out;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

CUresult __pool_cuEventDestroy(CUevent *hEvent, size_t count) {
  ava_async;
  ava_argument(hEvent) {
    ava_in;
    ava_buffer(count);
    ava_element ava_handle;
  }
}

/* AvA internal APIs */

void __do_batch_emit(void *command_buffer, size_t total_buffer_size) {
  ava_async;
  ava_argument(command_buffer) {
    ava_in;
    ava_buffer(total_buffer_size);
  }

  if (ava_is_worker) {
    // TODO: need to process return values
  }
}

/* APIs needed for a minimal program */

char CUDARTAPI __cudaInitModule(void **fatCubinHandle) {
  ava_argument(fatCubinHandle) {
    ava_in;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

ava_utility CUmodule __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle) {
  CUmodule mod = NULL;
  int ret;
  if (ava_metadata(NULL)->cuinit_called == 0) {
    ret = cuInit(0);
    if (ret != CUDA_SUCCESS) {
      fprintf(stderr, "cuInit fail: %d\n", ret);
    }
    ava_metadata(NULL)->cuinit_called = 1;
    assert(ret == CUDA_SUCCESS && "CUDA driver init failed");
    (void)ret;
  }
  __cudaInitModule(handle);
  ret = cuModuleLoadData(&mod, (void *)fatCubin->ptr);
  assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
  (void)ret;

  return mod;
}

ava_utility void __helper_load_function_arg_info_guest(void) {
  GPtrArray *fatbin_funcs;
  GHashTable *ht;
  if (ava_metadata(NULL)->fatbin_funcs == NULL) {
    ava_metadata(NULL)->fatbin_funcs = g_ptr_array_new_with_free_func(g_free);
    g_ptr_array_add(ava_metadata(NULL)->fatbin_funcs, (gpointer)NULL);  // func_id starts from 1
  }
  fatbin_funcs = ava_metadata(NULL)->fatbin_funcs;

  if (ava_metadata(NULL)->ht_name2idx == NULL) {
    ava_metadata(NULL)->ht_name2idx = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
  }
  ht = ava_metadata(NULL)->ht_name2idx;

  int fd, read_ret;
  char filename[50];
  sprintf(filename, "/cuda_dumps/function_arg-%d.ava", ava_metadata(NULL)->num_fatbins);
  AVA_DEBUG << "Loading " << filename;
  fd = open(filename, O_RDONLY, 0666);
  if (fd == -1) {
    fprintf(stderr, "open [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  struct fatbin_function *func;
  size_t name_size;
  char func_name[MAX_KERNEL_NAME_LEN];

  while (1) {
    read_ret = read(fd, (void *)&name_size, sizeof(size_t));
    if (read_ret == 0) break;
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    assert(name_size < MAX_KERNEL_NAME_LEN && "name_size >= MAX_KERNEL_NAME_LEN");
    read_ret = read(fd, (void *)func_name, name_size);
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    func = g_new(struct fatbin_function, 1);
    read_ret = read(fd, (void *)func, sizeof(struct fatbin_function));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    ava_debug("function %d (%s) has argc = %d", fatbin_funcs->len - 1, func_name, func->argc);
    /* Insert into the function table */
    g_ptr_array_add(fatbin_funcs, (gpointer)func);

    /* Add name->index mapping */
    if (g_hash_table_lookup(ht, func_name) == NULL) {
      assert(fatbin_funcs->len > 1 && "fatbin_funcs->len <= 1");
      g_hash_table_insert(ht, g_strdup(func_name), (gpointer)((uintptr_t)fatbin_funcs->len - 1));
    }
  }
  close(fd);

  ++(ava_metadata(NULL)->num_fatbins);
}

/**
 * Loads the function argument information from dump.
 */
ava_utility GHashTable *__helper_load_function_arg_info(void) {
  GPtrArray *fatbin_funcs;
  if (ava_metadata(NULL)->fatbin_funcs == NULL) {
    ava_metadata(NULL)->fatbin_funcs = g_ptr_array_new_with_free_func(g_free);
    g_ptr_array_add(ava_metadata(NULL)->fatbin_funcs, (gpointer)NULL);  // func_id starts from 1
  }
  fatbin_funcs = ava_metadata(NULL)->fatbin_funcs;

  GHashTable *ht = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);

  int fd, read_ret;
  char filename[50];
  sprintf(filename, "/cuda_dumps/function_arg-%d.ava", ava_metadata(NULL)->num_fatbins);
  AVA_DEBUG << "Loading " << filename;
  fd = open(filename, O_RDONLY, 0666);
  if (fd == -1) {
    fprintf(stderr, "open [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  struct fatbin_function *func;
  size_t name_size;
  char func_name[MAX_KERNEL_NAME_LEN];

  while (1) {
    read_ret = read(fd, (void *)&name_size, sizeof(size_t));
    if (read_ret == 0) break;
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    assert(name_size < MAX_KERNEL_NAME_LEN && "name_size >= MAX_KERNEL_NAME_LEN");
    read_ret = read(fd, (void *)func_name, name_size);
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    func = g_new(struct fatbin_function, 1);
    read_ret = read(fd, (void *)func, sizeof(struct fatbin_function));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    ava_debug("function %d (%s) has argc = %d", fatbin_funcs->len - 1, func_name, func->argc);
    /* Insert into the function table */
    g_ptr_array_add(fatbin_funcs, (gpointer)func);

    /* Add name->index mapping */
    if (g_hash_table_lookup(ht, func_name) == NULL) {
      assert(fatbin_funcs->len > 1 && "fatbin_funcs->len <= 1");
      g_hash_table_insert(ht, g_strdup(func_name), (gpointer)((uintptr_t)fatbin_funcs->len - 1));
    }
  }
  close(fd);

  ++(ava_metadata(NULL)->num_fatbins);
  return ht;
}

/**
 * This utility function should only be called by the worker.
 */
ava_utility void **__helper_load_and_register_fatbin(void *fatCubin) {
  /* Read fatbin dump */
  int fd, ret;
  int read_ret;
  struct stat file_stat;
  char filename[50];
  sprintf(filename, "/cuda_dumps/fatbin-%d.ava", ava_metadata(NULL)->num_fatbins);
  AVA_DEBUG << "Loading " << filename;
  fd = open(filename, O_RDONLY, 0666);
  if (fd == -1) {
    fprintf(stderr, "open [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  /* Create and read fatbin buffer */
  ret = fstat(fd, &file_stat);
  if (ret == -1) {
    fprintf(stderr, "fstat [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  size_t fatbin_size = (size_t)file_stat.st_size;
  void *fatbin = malloc(fatbin_size);
  if (fatbin == NULL) {
    fprintf(stderr, "malloc size=%lu [errno=%d, errstr=%s] at %s:%d", fatbin_size, errno, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  read_ret = read(fd, fatbin, fatbin_size);
  if (read_ret == -1) {
    fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  close(fd);

  struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)fatbin;
  ava_debug("Read fatbin-%d.ava size = %lu, should be %llu", ava_metadata(NULL)->num_fatbins, fatbin_size,
            fbh->headerSize + fbh->fatSize);
  assert(fatbin_size == fbh->headerSize + fbh->fatSize && "fatbin size is wrong");
  (void)fbh;

  /* Call native API to register the fatbin */
  struct fatbin_wrapper *wrapper = (struct fatbin_wrapper *)fatCubin;
  wrapper->ptr = (uint64_t)fatbin;

  void **fatbin_handle = __cudaRegisterFatBinary(wrapper);
  //__helper_print_fatcubin_info(fatCubin, fatbin_handle);
  CUmodule mod = __helper_init_module(wrapper, fatbin_handle);

  /* Load function argument information */
  GHashTable *ht = __helper_load_function_arg_info();

  /* Register CUDA functions */
  GPtrArray *fatbin_funcs = ava_metadata(NULL)->fatbin_funcs;
  struct fatbin_function *func;

  if (ava_metadata(NULL)->fatfunction_fd == 0) {
    ava_metadata(NULL)->fatfunction_fd = open("/cuda_dumps/fatfunction.ava", O_RDONLY, 0666);
  }
  fd = ava_metadata(NULL)->fatfunction_fd;

  void *func_id;
  size_t size;
  int exists;
  char *deviceFun;
  char *deviceName;
  int thread_limit;
  uint3 *tid;
  uint3 *bid;
  dim3 *bDim;
  dim3 *gDim;
  int *wSize;
  while (1) {
    read_ret = read(fd, (void *)&size, sizeof(size_t));
    if (read_ret == 0) {  // EOF
      close(fd);
      break;
    }
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    if (size == 0) {  // Meet separator
      ava_debug("Finish reading functions for fatbin-%d.ava", ava_metadata(NULL)->num_fatbins - 1);
      break;
    }
    deviceFun = (char *)malloc(size);
    if (deviceFun == NULL) {
      fprintf(stderr, "malloc size=0x%lx [errno=%d, errstr=%s] at %s:%d", size, errno, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    read_ret = read(fd, (void *)deviceFun, size);
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    read_ret = read(fd, (void *)&size, sizeof(size_t));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    deviceName = (char *)malloc(size);
    if (deviceName == NULL) {
      fprintf(stderr, "malloc [errno=%d, errstr=%s] at %s:%d, size=0x%lx", errno, strerror(errno), __FILE__, __LINE__,
              size);
      exit(EXIT_FAILURE);
    }
    read_ret = read(fd, (void *)deviceName, size);
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d, size=0x%lx", errno, strerror(errno), __FILE__, __LINE__,
              size);
      exit(EXIT_FAILURE);
    }

    read_ret = read(fd, (void *)&thread_limit, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    read_ret = read(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      tid = (uint3 *)malloc(sizeof(uint3));
      if (tid == NULL) {
        fprintf(stderr, "malloc size=%lu [errno=%d, errstr=%s] at %s:%d", sizeof(uint3), errno, strerror(errno),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      read_ret = read(fd, (void *)tid, sizeof(uint3));
      if (read_ret == -1) {
        fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      tid = NULL;

    read_ret = read(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      bid = (uint3 *)malloc(sizeof(uint3));
      if (bid == NULL) {
        fprintf(stderr, "malloc size=%lu [errno=%d, errstr=%s] at %s:%d", sizeof(uint3), errno, strerror(errno),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      read_ret = read(fd, (void *)bid, sizeof(uint3));
      if (read_ret == -1) {
        fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      bid = NULL;

    read_ret = read(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      bDim = (dim3 *)malloc(sizeof(dim3));
      read_ret = read(fd, (void *)bDim, sizeof(dim3));
      if (read_ret == -1) {
        fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      bDim = NULL;

    read_ret = read(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      gDim = (dim3 *)malloc(sizeof(dim3));
      if (gDim == NULL) {
        fprintf(stderr, "malloc size=%lu [errno=%d, errstr=%s] at %s:%d", sizeof(dim3), errno, strerror(errno),
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      read_ret = read(fd, (void *)gDim, sizeof(dim3));
      if (read_ret == -1) {
        fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      gDim = NULL;

    read_ret = read(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      wSize = (int *)malloc(sizeof(int));
      if (wSize == NULL) {
        fprintf(stderr, "malloc size=%lu [errno=%d, errstr=%s] at %s:%d", sizeof(int), errno, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
      read_ret = read(fd, (void *)wSize, sizeof(int));
      if (read_ret == -1) {
        fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      wSize = NULL;

    AVA_DEBUG << "Register function deviceName = " << deviceName;
    func_id = (void *)g_hash_table_lookup(ht, deviceName);
    assert(func_id != NULL && "func_id should not be NULL");
    func = static_cast<struct fatbin_function *>(g_ptr_array_index(fatbin_funcs, (intptr_t)func_id));
    __helper_register_function(func, (const char *)func_id, mod, deviceName);

    free(deviceFun);
    free(deviceName);
    if (tid) free(tid);
    if (bid) free(bid);
    if (bDim) free(bDim);
    if (gDim) free(gDim);
    if (wSize) free(wSize);
  }

  g_hash_table_destroy(ht);
  return fatbin_handle;
}

/*
void** CUDARTAPI
__cudaRegisterFatBinary(void *fatCubin)
{
    ava_disable_native_call;

    ava_argument(fatCubin) {
        ava_type_cast(struct fatbin_wrapper *);
        ava_in; ava_buffer(1);
        //ava_lifetime_static;
    }

    if (ava_is_guest) {
        ava_metadata(NULL)->ht_name2idx = __helper_load_function_arg_info();
    }

    if (ava_is_worker) {
        return __helper_load_and_register_fatbin((void *)fatCubin);
    }
    void **ret;
    ava_return_value {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
        ava_lifetime_manual;
    }
}
*/

/*
void** CUDARTAPI
__cudaRegisterFatBinary(void *fatCubin)
{
    ava_disable_native_call;

    ava_argument(fatCubin) ava_opaque;

    ava_implicit_argument
    int fatbin_num  = (ava_metadata(NULL)->fatbin_num_cur)++;

    if (ava_is_worker) {
        return (void **)g_ptr_array_index(fatbin_handle_list, fatbin_num);
    }

    void **ret;
    ava_return_value {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
        ava_lifetime_manual;
    }
}

void CUDARTAPI
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
    ava_disable_native_call;

    ava_argument(fatCubinHandle) {
        ava_in;
        ava_buffer(1); ava_element ava_handle;
    }

    if (ava_is_worker) {
        __helper_unregister_fatbin(fatCubinHandle);
    }
}
*/

ava_begin_replacement;
EXPORTED void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  void **dummy_fatbin = static_cast<void **>(malloc(sizeof(void *)));
  if (dummy_fatbin == NULL) {
    fprintf(stderr, "malloc size=%lu [errno=%d, errstr=%s] at %s:%d", sizeof(void *), errno, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  *dummy_fatbin = (void *)0x100;
  return dummy_fatbin;
}

EXPORTED void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
#warning Unregister fat binaries in guestlib and worker destruction code.
  return;
}
ava_end_replacement;

/**
 * Associate the local function pointer with the imported function ID.
 * This utility function should only be called in the guestlib.
 */
ava_utility void __helper_assosiate_function(void *local, const char *deviceName) {
  if (ava_metadata(local)->func != NULL) {
    ava_debug("Function (%s) metadata (%p) already exists, func_id = %p", deviceName, local,
              ava_metadata(local)->func_id);
    return;
  }

  void *func_id = (void *)g_hash_table_lookup(ava_metadata(NULL)->ht_name2idx, deviceName);
  if (func_id == NULL) {
    AVA_DEBUG << "DeviceName is " << deviceName;
    assert(0 && "func_id should not be null");
  }
  ava_metadata(local)->func_id = func_id;
  ava_metadata(local)->func =
      static_cast<struct fatbin_function *>(g_ptr_array_index(ava_metadata(NULL)->fatbin_funcs, (intptr_t)func_id));
  ava_debug("Function (%s) metadata (%p) is associated, func_id = %p", deviceName, local, ava_metadata(local)->func_id);
}

ava_begin_replacement;
EXPORTED void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                                               const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                                               dim3 *bDim, dim3 *gDim, int *wSize) {
  __helper_assosiate_function((void *)hostFun, deviceName);
}
ava_end_replacement;

ava_begin_replacement;
EXPORTED void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                                          const char *deviceName, int ext, size_t size, int constant, int global) {}

EXPORTED void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
#warning This API is called for CUDA 10.1 and 10.2, but it seems to be able to be ignored.
}
ava_end_replacement;

/*
__host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3   gridDim,
                            dim3   blockDim,
                            size_t sharedMem, // CHECKME: default argument in header
                            void   *stream)
{
    ava_async;
    ava_argument(stream) {
        ava_handle;
    }
}

cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3   *gridDim,
                           dim3   *blockDim,
                           size_t *sharedMem,
                           void   *stream)
{
    ava_argument(gridDim) {
        ava_out; ava_buffer(1);
    }
    ava_argument(blockDim) {
        ava_out; ava_buffer(1);
    }
    ava_argument(sharedMem) {
        ava_out; ava_buffer(1);
    }
    ava_argument(stream) {
        ava_type_cast(CUstream *);
        ava_out; ava_buffer(1);
        ava_element { ava_handle; }
    }
}
*/

ava_begin_replacement;
EXPORTED __host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                            size_t sharedMem,  // CHECKME: default argument in header
                            void *stream) {
  struct call_configuration *cc = static_cast<struct call_configuration *>(g_malloc(sizeof(struct call_configuration)));
  cc->gridDim = gridDim;
  cc->blockDim = blockDim;
  cc->sharedMem = sharedMem;
  cc->stream = stream;
  g_queue_push_tail(call_configuration_stack, (gpointer)cc);
  return 0;
}

EXPORTED cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem,
                                                          void *stream) {
  struct call_configuration *cc = static_cast<struct call_configuration *>(g_queue_pop_tail(call_configuration_stack));
  *gridDim = cc->gridDim;
  *blockDim = cc->blockDim;
  *sharedMem = cc->sharedMem;
  *(CUstream *)stream = (CUstream)cc->stream;
  g_free(cc);
  return cudaSuccess;
}
ava_end_replacement;

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                                                size_t sharedMem, cudaStream_t stream) {
  /* Cannot be ava_async, may lead to TensorFlow internal race condition */
  // ava_async;
  ava_disable_native_call;

  ava_implicit_argument void *func_id = ava_metadata(func)->func_id;
  ava_argument(func_id) { ava_opaque; }

  ava_argument(func) { ava_opaque; }

  ava_argument(args) {
#warning implicit arguments' dependency detection is broken.
    ava_depends_on(func_id);
    ava_in;
    // FIXME(athy): parser converts ava_metadata(NULL) to ava_metadata(()) used by g_ptr_array_index.
    ava_buffer(
        ((struct fatbin_function *)g_ptr_array_index(ava_metadata((void *)0)->fatbin_funcs, (intptr_t)func_id))->argc);
    ava_element {
      ava_type_cast(void *);
      ava_buffer(((struct fatbin_function *)g_ptr_array_index(ava_metadata((void *)0)->fatbin_funcs, (intptr_t)func_id))
                     ->args[__args_index_0]
                     .size);
      // ava_element ava_handle;
    }
  }

  ava_argument(stream) { ava_handle; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_launch_kernel(
        ((struct fatbin_function *)g_ptr_array_index(ava_metadata((void *)0)->fatbin_funcs, (intptr_t)func_id)),
        func_id, gridDim, blockDim, args, sharedMem, stream);
#warning This will bypass the resource reporting routine.
    return ret;
  }
}

ava_begin_replacement;
EXPORTED __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
  *ptr = malloc(size);
  if (ptr)
    return cudaSuccess;
  else
    return cudaErrorMemoryAllocation;
}

EXPORTED __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  free(ptr);
  return cudaSuccess;
}
ava_end_replacement;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  ava_argument(devPtr) {
    ava_out;
    ava_buffer(1);
    ava_element ava_opaque;
  }
}

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  ava_argument(dst) {
    if (kind == cudaMemcpyHostToDevice) {
      ava_opaque;
    } else if (kind == cudaMemcpyDeviceToHost) {
      ava_out;
      ava_buffer(count);
    }
  }

  ava_argument(src) {
    if (kind == cudaMemcpyHostToDevice) {
      ava_in;
      ava_buffer(count);
    } else if (kind == cudaMemcpyDeviceToHost) {
      ava_opaque;
    }
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr) { ava_argument(devPtr) ava_opaque; }

/* Rich set of APIs */

cudaError_t CUDARTAPI cudaLaunch(const void *func) { ava_unsupported; }

cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset) { ava_unsupported; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  ava_argument(device) {
    ava_out;
    ava_buffer(1);
  }
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  ava_argument(count) {
    ava_out;
    ava_buffer(1);
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  ava_argument(prop) {
    ava_out;
    ava_buffer(1);
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr,
                                                                         int device) {
  ava_argument(value) {
    ava_out;
    ava_buffer(1);
  }
}

__host__ cudaError_t CUDARTAPI cudaDeviceReset(void);

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device);

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset,
                                                  enum cudaMemcpyKind kind) {
  ava_argument(symbol) { ava_opaque; }
  ava_argument(src) {
    ava_in;
    ava_buffer(count);
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                                                  enum cudaMemcpyKind kind, cudaStream_t stream) {
  /* TensorFlow always copies data between device memories */
  ava_async;
  ava_argument(dst) ava_opaque;
  ava_argument(src) ava_opaque;

  /*
  ava_argument(dst) {
      if (kind == cudaMemcpyHostToDevice) {
          ava_opaque;
      }
      else if (kind == cudaMemcpyDeviceToHost) {
          ava_out; ava_buffer(count);
      }
  }

  ava_argument(src) {
      if (kind == cudaMemcpyHostToDevice) {
          ava_in; ava_buffer(count);
      }
      else if (kind == cudaMemcpyDeviceToHost) {
          ava_opaque;
      }
  }
  */

  ava_argument(stream) ava_handle;

#warning Force synchronization of async buffers
  ava_execute();
  if (ava_is_worker && kind == cudaMemcpyDeviceToHost) {
    cudaStreamSynchronize(stream);
  }
}

__host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count) { ava_argument(devPtr) ava_opaque; }

/*
__host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr)
{
    ava_argument(attributes) {
        ava_out; ava_buffer(1);
    }
    ava_argument(ptr) {
        //ava_type_cast(CUdeviceptr);
        //ava_handle;
        ava_opaque;
    }

    //__helper_print_pointer_attributes(attributes, ptr);
}
*/

ava_utility gint gpu_address_search_func(gconstpointer a, gconstpointer b) {
  struct gpu_address_range *r = (struct gpu_address_range *)g_tree_lookup(gpu_address_set, a);
  if (r->start > (uintptr_t)b) return -1;
  if (r->end <= (uintptr_t)b) return 1;
  return 0;
}

ava_begin_replacement;
EXPORTED __host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                                                 const void *ptr) {
  if (!attributes) return cudaErrorInvalidDevice;

  /* Search in gpu_address_set */
  gpointer res = g_tree_search(gpu_address_set, gpu_address_search_func, (gconstpointer)ptr);
  if (res) {
    attributes->type = cudaMemoryTypeDevice;  // maybe cudaMemoryTypeManaged?
    attributes->memoryType = cudaMemoryTypeDevice;
    return cudaSuccess;
  }

  attributes->type = cudaMemoryTypeUnregistered;
  attributes->memoryType = cudaMemoryTypeUnregistered;
  cuda_last_error = cudaErrorInvalidValue;
  return cudaErrorInvalidValue;
}
ava_end_replacement;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void);

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  ava_argument(event) ava_handle;
  ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) { ava_argument(event) ava_handle; }

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  ava_argument(ms) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(start) ava_handle;
  ava_argument(end) ava_handle;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) { ava_argument(event) ava_handle; }

/*
ava_callback_decl void __callback_cuda_stream_add_callback(
        cudaStream_t stream,  cudaError_t status, void*  userData) {
    ava_argument(stream) ava_handle;
    ava_argument(userData) {
        ava_userdata;
    }
}

__host__ cudaError_t CUDARTAPI
cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags)
{
    ava_argument(stream) ava_handle;
    ava_argument(callback) ava_callback(__callback_cuda_stream_add_callback);
}
*/

ava_begin_replacement;
EXPORTED __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback,
                                                              void *userData, unsigned int flags) {
#warning TODO: Fix callback.
  return cudaSuccess;
}

EXPORTED __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  cudaError_t ret = cuda_last_error;
  cuda_last_error = cudaSuccess;
  return ret;
}
ava_end_replacement;

__host__ __cudart_builtin__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}

__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *_free, size_t *total) {
  ava_argument(_free) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(total) {
    ava_out;
    ava_buffer(1);
  }
}

/* CUDA driver API */

CUresult CUDAAPI cuInit(unsigned int Flags) {
  ava_disable_native_call;

  if (ava_is_worker) {
    CUresult ret = CUDA_SUCCESS;
    if (ava_metadata(NULL)->cuinit_called == 0) {
      ret = cuInit(Flags);
      ava_metadata(NULL)->cuinit_called = 1;
    }
    return ret;
  }
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  ava_argument(hfunc) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(name) {
    ava_in;
    ava_buffer(strlen(name) + 1);
  }

  ava_execute();
  __helper_parse_function_args(name, ava_metadata(*hfunc)->func->args);
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  ava_argument(module) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(image) {
    ava_in;
    ava_buffer(__helper_fatbin_size(image));
  }
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) { ava_unsupported; }

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  ava_argument(hStream) ava_handle;

  ava_argument(kernelParams) {
    ava_in;
    ava_buffer(ava_metadata(f)->func->argc);
    ava_element {
      // FIXME: use the generated index name in the spec to
      // reference the outer loop's loop index at this moment.
      if (ava_metadata(f)->func->args[__kernelParams_index_0].is_handle) {
        ava_type_cast(void *);
        ava_buffer(ava_metadata(f)->func->args[__kernelParams_index_0].size);
        ava_element ava_opaque;
      } else {
        ava_type_cast(void *);
        ava_buffer(ava_metadata(f)->func->args[__kernelParams_index_0].size);
      }
    }
  }

  ava_argument(extra) {
    ava_in;
    ava_buffer(__helper_launch_extra_size(extra));
#warning The buffer size below states that every kernelParams[i] is 1 byte long.
    ava_element ava_buffer(1);
  }
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
  ava_argument(count) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  ava_argument(device) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
  ava_argument(canAccessPeer) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(dev) ava_handle;
  ava_argument(peerDev) ava_handle;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
  ava_argument(peerContext) ava_handle;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  ava_argument(device) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  ava_argument(name) {
    ava_out;
    ava_buffer(len);
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  ava_argument(uuid) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  ava_argument(pi) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
  ava_argument(flags) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(active) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) { ava_argument(dev) ava_handle; }

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  ava_argument(pctx) {
    ava_out;
    ava_element(ava_allocates);
    ava_buffer(1);
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) { ava_argument(ctx) ava_deallocates; }

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
  ava_argument(pctx) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) { ava_argument(ctx) ava_handle; }

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  ava_argument(pctx) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) { ava_argument(dev) ava_handle; }

CUresult CUDAAPI cuCtxSynchronize(void);

CUresult cuCtxPushCurrent(CUcontext ctx) { ava_argument(ctx) ava_handle; }

CUresult cuCtxPopCurrent(CUcontext *pctx) {
  ava_argument(pctx) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) { ava_unsupported; }

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) { ava_unsupported; }

CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) { ava_unsupported; }

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  ava_argument(phStream) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
  ava_argument(hStream) ava_handle;

  ava_argument(pctx) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
  ava_unsupported;
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) { ava_argument(hStream) ava_handle; }

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) { ava_argument(hStream) ava_handle; }

ava_utility void __helper_save_gpu_address_range(CUdeviceptr *dptr, size_t bytesize, void *ret) {
  if (ava_is_guest) {
    CUresult *cu_ret = static_cast<CUresult *>(ret);
    if (cu_ret != nullptr && *cu_ret == CUDA_SUCCESS) {
      struct gpu_address_range *range = (struct gpu_address_range *)g_malloc(sizeof(struct gpu_address_range));
      range->start = (uintptr_t)*dptr;
      range->end = (uintptr_t)*dptr + bytesize;
      g_tree_insert(gpu_address_set, (gpointer)range->start, (gpointer)range);
      ava_debug("Save GPU address range [%lx, %lx)", range->start, range->end);
    }
  }
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  ava_argument(dptr) {
    ava_out;
    ava_buffer(1);
    ava_element {
      ava_opaque;
      ava_allocates;
    }
  }

  void *ret = reinterpret_cast<void *>(ava_execute());
  __helper_save_gpu_address_range(dptr, bytesize, static_cast<void *>(&ret));
}

/*
CUresult CUDAAPI
cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
{
    ava_argument(pp) {
        ava_out; ava_buffer(1);
        ava_element {
            ava_buffer(bytesize);
            ava_buffer_allocator(__helper_cu_mem_host_alloc_portable,
                    __helper_cu_mem_host_free);
            ava_lifetime_manual;
            ava_allocates;
            ava_no_copy;
        }
    }

    ava_execute();
    ava_metadata(*pp)->is_pinned = 1;
}
*/

ava_begin_replacement;
EXPORTED CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
  *pp = __helper_cu_mem_host_alloc_portable(bytesize);
  return (*pp) ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
ava_end_replacement;

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
  ava_argument(dstDevice) ava_opaque;

  ava_argument(srcHost) {
    ava_in;
    ava_buffer(ByteCount);
    if (ava_metadata(srcHost)->is_pinned) ava_lifetime_manual;
  }
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  ava_argument(dstHost) {
    ava_out;
    ava_buffer(ByteCount);
    if (ava_metadata(dstHost)->is_pinned) ava_lifetime_manual;
  }

  ava_argument(srcDevice) ava_opaque;
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
  ava_async;
  ava_argument(dstDevice) ava_opaque;

  ava_argument(srcHost) {
    ava_in;
    ava_buffer(ByteCount);
    // if (ava_metadata(srcHost)->is_pinned) {
    ava_lifetime_manual;
    //}
    // else {
    //    ava_lifetime_manual;
    //}
#warning[issue#65] deallocate the buffer for async memory copy at the \
        synchronization point (ava_lifetime_sync).
  }

  ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  /*
  __helper_register_async_buffer(&ava_metadata(hStream)->async_buffers,
                              dstHost, ByteCount);
  */

  ava_argument(dstHost) {
#warning async buffers need to be no_copy
    // ava_no_copy;
    ava_out;
    ava_buffer(ByteCount);
    // if (ava_metadata(dstHost)->is_pinned) {
    //    ava_lifetime_manual;
    //}
    // else {
    //    ava_lifetime_manual;
    //}
#warning[issue#65] deallocate the buffer for async memory copy at the \
        synchronization point (ava_lifetime_sync).
  }

  ava_argument(srcDevice) ava_opaque;
  ava_argument(hStream) ava_handle;

#warning Force synchronization of async buffers
  ava_execute();
  if (ava_is_worker) {
    cudaStreamSynchronize(hStream);
  }
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  ava_async;
  ava_argument(dstDevice) ava_opaque;
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  ava_async;
  ava_argument(dstDevice) ava_opaque;
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
  ava_async;
  ava_argument(dstDevice) ava_opaque;
  ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
  ava_async;
  ava_argument(dstDevice) ava_opaque;
  ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI cuMemFreeHost(void *p) {
  ava_metadata(p)->is_pinned = 0;
  ava_deallocates;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  ava_argument(driverVersion) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) { ava_unsupported; }

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  ava_argument(bytes) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(dev) ava_handle;
}

CUresult CUDAAPI cuMemGetInfo(size_t *_free, size_t *total) {
  ava_argument(_free) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(total) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  ava_argument(pciBusId) {
    ava_out;
    ava_buffer(len);
  }
  ava_argument(dev) ava_handle;
}

ava_begin_replacement;
EXPORTED CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  CUresult res = CUDA_SUCCESS;

  if (g_queue_is_empty(cu_event_pool)) {
    size_t count = DESCRITPOR_POOL_SIZE;
    CUevent *desc = (CUevent *)malloc(sizeof(CUevent) * count);
    int i;
    res = __pool_cuEventCreate(desc, count);

    if (res == CUDA_SUCCESS) {
      for (i = 0; i < count; i++) g_queue_push_tail(cu_event_pool, (gpointer)desc[i]);
    }
  }

  if (res != CUDA_SUCCESS) return res;

  *phEvent = (CUevent)g_queue_pop_head(cu_event_pool);
  return res;
}
ava_end_replacement;

/*
CUresult CUDAAPI
cuEventQuery(CUevent hEvent)
{
    ava_argument(hEvent) ava_handle;
}
*/

CUresult __cuEventQuery(CUevent hEvent) {
  ava_async;
  ava_argument(hEvent) ava_handle;
}

ava_begin_replacement;
EXPORTED CUresult CUDAAPI cuEventQuery(CUevent hEvent) { return __cuEventQuery(hEvent); }
ava_end_replacement;

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  ava_async;
  ava_argument(hEvent) ava_handle;
  ava_argument(hStream) ava_handle;
}

/*
CUresult CUDAAPI
cuEventSynchronize(CUevent hEvent) {
    ava_argument(hEvent) ava_handle;
}

CUresult CUDAAPI
cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
    ava_argument(pMilliseconds) {
        ava_out; ava_buffer(1);
    }
    ava_argument(hStart) ava_handle;
    ava_argument(hEnd) ava_handle;
}
*/

ava_begin_replacement;
EXPORTED CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) { return CUDA_SUCCESS; }

EXPORTED CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
  *pMilliseconds = 10.0;
  return CUDA_SUCCESS;
}
ava_end_replacement;

ava_begin_replacement;
EXPORTED CUresult cuEventDestroy(CUevent hEvent) {
  g_queue_push_tail(idle_cu_event_pool, (gpointer)hEvent);
  if (idle_cu_event_pool->length >= DESCRITPOR_POOL_SIZE) return (CUresult)free_cu_event_pool(idle_cu_event_pool);
  return CUDA_SUCCESS;
}
ava_end_replacement;

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  /*
#warning Fix the update of the buffers that are copied asynchronously.
  ava_implicit_argument
  struct async_buffer_list *async_buffers = __helper_load_async_buffer_list(
          &ava_metadata(hStream)->async_buffers);
  ava_argument(async_buffers) {
      ava_out; ava_buffer(1);
  }
  */

  ava_async;
  ava_argument(hStream) ava_handle;
  ava_argument(hEvent) ava_handle;
}

CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) { ava_unsupported; }

CUresult cuGetErrorName(CUresult error, const char **pStr) {
  ava_argument(pStr) {
    ava_out;
    ava_buffer(1);
    ava_element {
      ava_lifetime_manual;
      ava_buffer(100);
    }
  }
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize,
                                                             size_t dynamicSMemSize) {
  ava_argument(numBlocks) {
    ava_out;
    ava_buffer(1);
  }

  ava_unsupported;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize,
                                                                      size_t dynamicSMemSize, unsigned int flags) {
  ava_argument(numBlocks) {
    ava_out;
    ava_buffer(1);
  }

  ava_unsupported;
}

/* CUDABLAS API */

#include "cava/samples/cuda_common_spec/cublas/cublas.h"

/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                     cublasOperation_t transb, int m, int n, int k,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A, int lda, const float *B, int ldb,
                                                     const float *beta, /* host or device pointer */
                                                     float *C, int ldc) {
  ava_async;
  ava_argument(handle) ava_handle;

  ava_argument(transa) ava_opaque;
  ava_argument(transb) ava_opaque;
  /* These are always device pointers for tensorflow ! */
  ava_argument(A) ava_opaque;
  ava_argument(B) ava_opaque;
  ava_argument(C) ava_opaque;

  ava_argument(alpha) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_in;
    ava_buffer(1);
  }

  /*
#warning Force synchronization of async buffers
  ava_execute();
  if (ava_is_worker) {
      cudaStream_t streamId;
      cublasGetStream(handle, &streamId);
      cudaStreamSynchronize(streamId);
  }
  */
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy(cublasHandle_t handle) { ava_argument(handle) ava_handle; }

/***** CUDNN (OOF) ******/

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,                                                   /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x,                 /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,                       /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance, double epsilon) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(yDesc) ava_handle;
  ava_argument(y) ava_opaque;
  ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
  ava_argument(bnScale) ava_opaque;
  ava_argument(bnBias) ava_opaque;
  ava_argument(estimatedMean) ava_opaque;
  ava_argument(estimatedVariance) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                                                  const cudnnTensorDescriptor_t xDesc, const void *x,
                                                  const cudnnFilterDescriptor_t wDesc, const void *w,
                                                  const cudnnConvolutionDescriptor_t convDesc,
                                                  cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                  size_t workSpaceSizeInBytes, const void *beta,
                                                  const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(wDesc) ava_handle;
  ava_argument(w) ava_opaque;
  ava_argument(convDesc) ava_handle;
  ava_argument(workSpace) ava_opaque;
  ava_argument(yDesc) ava_handle;
  ava_argument(y) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
  ava_argument(handle) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) { ava_argument(handle) ava_handle; }

ava_begin_replacement;
EXPORTED cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  if (g_queue_is_empty(convolution_descriptor_pool)) {
    size_t count = DESCRITPOR_POOL_SIZE;
    cudnnConvolutionDescriptor_t *desc =
        (cudnnConvolutionDescriptor_t *)malloc(sizeof(cudnnConvolutionDescriptor_t) * count);
    int i;
    res = __pool_cudnnCreateConvolutionDescriptor(desc, count);

    if (res == CUDNN_STATUS_SUCCESS) {
      for (i = 0; i < count; i++) g_queue_push_tail(convolution_descriptor_pool, (gpointer)desc[i]);
    }
  }

  if (res != CUDNN_STATUS_SUCCESS) return res;

  *convDesc = (cudnnConvolutionDescriptor_t)g_queue_pop_head(convolution_descriptor_pool);
  return res;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  if (g_queue_is_empty(filter_descriptor_pool)) {
    size_t count = DESCRITPOR_POOL_SIZE;
    cudnnFilterDescriptor_t *desc = (cudnnFilterDescriptor_t *)malloc(sizeof(cudnnFilterDescriptor_t) * count);
    int i;
    res = __pool_cudnnCreateFilterDescriptor(desc, count);

    if (res == CUDNN_STATUS_SUCCESS) {
      for (i = 0; i < count; i++) g_queue_push_tail(filter_descriptor_pool, (gpointer)desc[i]);
    }
  }

  if (res != CUDNN_STATUS_SUCCESS) return res;

  *filterDesc = (cudnnFilterDescriptor_t)g_queue_pop_head(filter_descriptor_pool);
  return res;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  if (g_queue_is_empty(pooling_descriptor_pool)) {
    size_t count = DESCRITPOR_POOL_SIZE;
    cudnnPoolingDescriptor_t *desc = (cudnnPoolingDescriptor_t *)malloc(sizeof(cudnnPoolingDescriptor_t) * count);
    int i;
    res = __pool_cudnnCreatePoolingDescriptor(desc, count);

    if (res == CUDNN_STATUS_SUCCESS) {
      for (i = 0; i < count; i++) g_queue_push_tail(pooling_descriptor_pool, (gpointer)desc[i]);
    }
  }

  if (res != CUDNN_STATUS_SUCCESS) return res;

  *poolingDesc = (cudnnPoolingDescriptor_t)g_queue_pop_head(pooling_descriptor_pool);
  return res;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  if (g_queue_is_empty(tensor_descriptor_pool)) {
    size_t count = DESCRITPOR_POOL_SIZE;
    cudnnTensorDescriptor_t *desc = (cudnnTensorDescriptor_t *)malloc(sizeof(cudnnTensorDescriptor_t) * count);
    int i;
    res = __pool_cudnnCreateTensorDescriptor(desc, count);

    if (res == CUDNN_STATUS_SUCCESS) {
      for (i = 0; i < count; i++) g_queue_push_tail(tensor_descriptor_pool, (gpointer)desc[i]);
    }
  }

  if (res != CUDNN_STATUS_SUCCESS) return res;

  *tensorDesc = (cudnnTensorDescriptor_t)g_queue_pop_head(tensor_descriptor_pool);
  return res;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  g_queue_push_tail(idle_convolution_descriptor_pool, (gpointer)convDesc);
  if (idle_convolution_descriptor_pool->length >= DESCRITPOR_POOL_SIZE)
    return (cudnnStatus_t)free_convolution_descriptor_pool(idle_convolution_descriptor_pool);
  return CUDNN_STATUS_SUCCESS;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  g_queue_push_tail(idle_filter_descriptor_pool, (gpointer)filterDesc);
  if (idle_filter_descriptor_pool->length >= DESCRITPOR_POOL_SIZE)
    return (cudnnStatus_t)free_filter_descriptor_pool(idle_filter_descriptor_pool);
  return CUDNN_STATUS_SUCCESS;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  g_queue_push_tail(idle_pooling_descriptor_pool, (gpointer)poolingDesc);
  if (idle_pooling_descriptor_pool->length >= DESCRITPOR_POOL_SIZE)
    return (cudnnStatus_t)free_pooling_descriptor_pool(idle_pooling_descriptor_pool);
  return CUDNN_STATUS_SUCCESS;
}

EXPORTED cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  g_queue_push_tail(idle_tensor_descriptor_pool, (gpointer)tensorDesc);
  if (idle_tensor_descriptor_pool->length >= DESCRITPOR_POOL_SIZE)
    return (cudnnStatus_t)free_tensor_descriptor_pool(idle_tensor_descriptor_pool);
  return CUDNN_STATUS_SUCCESS;
}
ava_end_replacement;

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(zDesc) ava_handle;
  ava_argument(yDesc) ava_handle;
  ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
  ava_argument(activationDesc) ava_handle;
  ava_argument(sizeInBytes) {
    ava_out;
    ava_buffer(1);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                                                  const cudnnTensorDescriptor_t xDesc,
                                                                  const cudnnFilterDescriptor_t wDesc,
                                                                  const cudnnConvolutionDescriptor_t convDesc,
                                                                  const cudnnTensorDescriptor_t yDesc,
                                                                  cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(wDesc) ava_handle;
  ava_argument(convDesc) ava_handle;
  ava_argument(yDesc) ava_handle;
  ava_argument(sizeInBytes) {
    ava_out;
    ava_buffer(1);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionFwdAlgo_t *algo) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(wDesc) ava_handle;
  ava_argument(convDesc) ava_handle;
  ava_argument(yDesc) ava_handle;
  ava_argument(algo) {
    ava_out;
    ava_buffer(1);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(wDesc) ava_handle;
  ava_argument(convDesc) ava_handle;
  ava_argument(yDesc) ava_handle;
  ava_argument(returnedAlgoCount) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(perfResults) {
    ava_out;
    cu_in_out_buffer(requestedAlgoCount, returnedAlgoCount);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int *value) {
  ava_argument(value) {
    ava_out;
    ava_buffer(1);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                              const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                              const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(poolingDesc) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(yDesc) ava_handle;
  ava_argument(y) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  ava_async;
  ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
  ava_async;
  ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                          int arrayLength, /* nbDims-2 size */
                                                          const int padA[], const int filterStrideA[],
                                                          const int dilationA[], cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t computeType) /* convolution data type */
{
  ava_async;
  ava_argument(convDesc) ava_handle;
  ava_argument(padA) {
    ava_in;
    ava_buffer(arrayLength);
  }
  ava_argument(filterStrideA) {
    ava_in;
    ava_buffer(arrayLength);
  }
  ava_argument(dilationA) {
    ava_in;
    ava_buffer(arrayLength);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                     cudnnDataType_t dataType, /* image data type */
                                                     cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
  ava_async;
  ava_argument(filterDesc) ava_handle;
  ava_argument(filterDimA) {
    ava_in;
    ava_buffer(nbDims);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                      const cudnnPoolingMode_t mode,
                                                      const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
                                                      const int windowDimA[], const int paddingA[],
                                                      const int strideA[]) {
  ava_async;
  ava_argument(poolingDesc) ava_handle;
  ava_argument(windowDimA) {
    ava_in;
    ava_buffer(nbDims);
  }
  ava_argument(paddingA) {
    ava_in;
    ava_buffer(nbDims);
  }
  ava_argument(strideA) {
    ava_in;
    ava_buffer(nbDims);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(streamId) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
                                                     int nbDims, const int dimA[], const int strideA[]) {
  ava_async;
  ava_argument(tensorDesc) ava_handle;
  ava_argument(dimA) {
    ava_in;
    ava_buffer(nbDims);
  }
  ava_argument(strideA) {
    ava_in;
    ava_buffer(nbDims);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                               const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
                                               const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                               const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                               const cudnnTensorDescriptor_t dxDesc, void *dx) {
  ava_argument(handle) ava_handle;
  ava_argument(poolingDesc) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(yDesc) ava_handle;
  ava_argument(y) ava_opaque;
  ava_argument(dyDesc) ava_handle;
  ava_argument(dy) ava_opaque;
  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(dxDesc) ava_handle;
  ava_argument(dx) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  ava_argument(rnnDesc) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) { ava_argument(rnnDesc) ava_handle; }

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                const int hiddenSize, const int numLayers,
                                                cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
                                                cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize,
                                                int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
                                                cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction,
                                                cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                          cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt, double lclip,
                                          double rclip) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                          cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt,
                                          double *lclip, double *rclip) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                      const int recProjSize, const int outProjSize) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                      int *recProjSize, int *outProjSize) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

/* Expensive. Creates the plan for the specific settings. */
cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, const int minibatch,
                                                       const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

/* dataType in weight descriptors and input descriptors is used to describe storage */
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength, const cudnnTensorDescriptor_t *xDesc,
                                                   size_t *sizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                         const int seqLength, const cudnnTensorDescriptor_t *xDesc,
                                                         size_t *sizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
                                                cudnnDataType_t dataType) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                          const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
                                                          const cudnnFilterDescriptor_t wDesc, const void *w,
                                                          const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
                                                          void **linLayerMat) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                        const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
                                                        const cudnnFilterDescriptor_t wDesc, const void *w,
                                                        const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
                                                        void **linLayerBias) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                  const int seqLength, const cudnnTensorDescriptor_t *xDesc,
                                                  const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx,
                                                  const cudnnTensorDescriptor_t *yDesc, const void *y,
                                                  const void *workspace, size_t workSpaceSizeInBytes,
                                                  const cudnnFilterDescriptor_t dwDesc, void *dw,
                                                  const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

/* RNN EX API */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType, cudnnRNNDataLayout_t layout, int maxSeqLength,
    int batchSize, int vectorSize, const int seqLengthArray[], /* length of each sequence in the batch */
    void *paddingFill)                                         /* symbol for filling padding position in output */
{
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
                                                    cudnnRNNDataLayout_t *layout, int *maxSeqLength, int *batchSize,
                                                    int *vectorSize, int arrayLengthRequested, int seqLengthArray[],
                                                    void *paddingFill) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
    const void *keys,                     /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
    void *cAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
    void *iAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
    void *queries,                        /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInferenceEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
    const void *keys,                     /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
    void *cAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
    void *iAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
    void *queries,                        /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardDataEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, /* reserved, should pass NULL */
    const void *dcAttn,                    /* reserved, should pass NULL */
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, /* reserved, should pass NULL */
    void *dkeys,                           /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeightsEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                                                    const cudnnRNNDataDescriptor_t xDesc, const void *x,
                                                    const cudnnTensorDescriptor_t hxDesc, const void *hx,
                                                    const cudnnRNNDataDescriptor_t yDesc, const void *y,
                                                    void *workSpace, size_t workSpaceSizeInBytes,
                                                    const cudnnFilterDescriptor_t dwDesc, void *dw, void *reserveSpace,
                                                    size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

/* RNN FIND API */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                         cudnnAlgorithmDescriptor_t algoDesc) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle,
                                                                       const cudnnRNNDescriptor_t rnnDesc, int *count) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle,
                                                                      const cudnnRNNDescriptor_t rnnDesc, int *count) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                                   const cudnnRNNDescriptor_t rnnDesc, int *count) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle,
                                                                      const cudnnRNNDescriptor_t rnnDesc, int *count) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

/* DEPRECATED routines to be removed next release :
   User should use the non-suffixed version (which has the API and functionality of _v6 version)
   Routines with _v5 suffix has the functionality of the non-suffixed routines in the CUDNN V6
 */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                   const int hiddenSize, const int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                   cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                   cudnnDataType_t mathPrec) {
  fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
                                                     cudnnDataType_t dataType, /* image data type */
                                                     int n,                    /* number of inputs (batch size) */
                                                     int c,                    /* number of input feature maps */
                                                     int h,                    /* height of input section */
                                                     int w)                    /* width of input section */
{
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                       cudnnDataType_t dataType, /* image data type */
                                                       int n,                    /* number of inputs (batch size) */
                                                       int c,                    /* number of input feature maps */
                                                       int h,                    /* height of input section */
                                                       int w,                    /* width of input section */
                                                       int nStride, int cStride, int hStride, int wStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                                                     cudnnDataType_t *dataType, /* image data type */
                                                     int *n,                    /* number of inputs (batch size) */
                                                     int *c,                    /* number of input feature maps  */
                                                     int *h,                    /* height of input section */
                                                     int *w,                    /* width of input section */
                                                     int *nStride, int *cStride, int *hStride, int *wStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
                                                     cudnnDataType_t *dataType, int *nbDims, int dimA[],
                                                     int strideA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
  ava_unsupported;
}

/** Create a destination descriptor for cudnnTransformTensor */
cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                                                 const cudnnTensorDescriptor_t srcDesc,
                                                 cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes) {
  ava_unsupported;
}

/** Create an empty tensor transform descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc) {
  ava_unsupported;
}

/** Initialize a previously created tensor transform descriptor. */
cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                            const uint32_t nbDims, const cudnnTensorFormat_t destFormat,
                                                            const int32_t padBeforeA[], const int32_t padAfterA[],
                                                            const uint32_t foldA[],
                                                            const cudnnFoldingDirection_t direction) {
  ava_unsupported;
}

/**
 * Retrieves the values stored in a previously initialized tensor transform
 * descriptor.
 */
cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                            uint32_t nbDimsRequested, cudnnTensorFormat_t *destFormat,
                                                            int32_t padBeforeA[], int32_t padAfterA[], uint32_t foldA[],
                                                            cudnnFoldingDirection_t *direction) {
  ava_unsupported;
}

/**
 * Destroys a previously created tensor transform descriptor.
 */
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc) {
  ava_unsupported;
}

/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle, const void *alpha,
                                               const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                               const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
                                                 const void *alpha, const cudnnTensorDescriptor_t srcDesc,
                                                 const void *srcData, const void *beta,
                                                 const cudnnTensorDescriptor_t destDesc, void *destData) {
  ava_unsupported;
}

/* Helper function to calculate folding descriptors  for dgrad */
cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat, cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc, cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc, cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc, cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
  ava_unsupported;
}

/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t aDesc,
                                         const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc,
                                         void *C) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                                     cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType,
                                                     cudnnNanPropagation_t opTensorNanOpt) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                                                     cudnnOpTensorOp_t *opTensorOp, cudnnDataType_t *opTensorCompType,
                                                     cudnnNanPropagation_t *opTensorNanOpt) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) { ava_unsupported; }

/* Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/* B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
cudnnStatus_t CUDNNWINAPI cudnnOpTensor(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
                                        const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
                                        const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
                                        const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         cudnnReduceTensorOp_t reduceTensorOp,
                                                         cudnnDataType_t reduceTensorCompType,
                                                         cudnnNanPropagation_t reduceTensorNanOpt,
                                                         cudnnReduceTensorIndices_t reduceTensorIndices,
                                                         cudnnIndicesType_t reduceTensorIndicesType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         cudnnReduceTensorOp_t *reduceTensorOp,
                                                         cudnnDataType_t *reduceTensorCompType,
                                                         cudnnNanPropagation_t *reduceTensorNanOpt,
                                                         cudnnReduceTensorIndices_t *reduceTensorIndices,
                                                         cudnnIndicesType_t *reduceTensorIndicesType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  ava_unsupported;
}

/* Helper function to return the minimum size of the index space to be passed to the reduction given the input and
 * output tensors */
cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                                       const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                       const cudnnTensorDescriptor_t aDesc,
                                                       const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

/* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output
 * tensors */
cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                                         const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         const cudnnTensorDescriptor_t aDesc,
                                                         const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

/* Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                            void *indices, size_t indicesSizeInBytes, void *workspace,
                                            size_t workspaceSizeInBytes, const void *alpha,
                                            const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
                                            const cudnnTensorDescriptor_t cDesc, void *C) {
  ava_unsupported;
}

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                         const void *valuePtr) {
  ava_unsupported;
}

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                           const void *alpha) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                     cudnnDataType_t dataType, /* image data type */
                                                     cudnnTensorFormat_t format,
                                                     int k, /* number of output feature maps */
                                                     int c, /* number of input feature maps */
                                                     int h, /* height of each input filter */
                                                     int w) {
  ava_unsupported;
} /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                                                     cudnnDataType_t *dataType, /* image data type */
                                                     cudnnTensorFormat_t *format,
                                                     int *k, /* number of output feature maps */
                                                     int *c, /* number of input feature maps */
                                                     int *h, /* height of each input filter */
                                                     int *w) {
  ava_unsupported;
} /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
                                                     cudnnDataType_t *dataType, /* image data type */
                                                     cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
  ava_unsupported;
}
cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnTransformFilter(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
                                               const void *alpha, const cudnnFilterDescriptor_t srcDesc,
                                               const void *srcData, const void *beta,
                                               const cudnnFilterDescriptor_t destDesc, void *destData) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
                                                    cudnnReorderType_t reorderType, const void *filterData,
                                                    void *reorderedFilterData, int reorderBias, const void *biasData,
                                                    void *reorderedBiasData) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnMathType_t *mathType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                         cudnnReorderType_t reorderType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                         cudnnReorderType_t *reorderType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h, /* zero-padding height */
                                int pad_w,                                        /* zero-padding width */
                                int u,                                            /* vertical filter stride */
                                int v,                                            /* horizontal filter stride */
                                int dilation_h, /* filter dilation in the vertical dimension */
                                int dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc, int *pad_h, /* zero-padding height */
                                int *pad_w,                                              /* zero-padding width */
                                int *u,                                                  /* vertical filter stride */
                                int *v,                                                  /* horizontal filter stride */
                                int *dilation_h, /* filter dilation in the vertical dimension */
                                int *dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  ava_unsupported;
}

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                                const cudnnFilterDescriptor_t filterDesc, int *n,
                                                                int *c, int *h, int *w) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                                          int arrayLengthRequested, int *arrayLength, int padA[],
                                                          int strideA[], int dilationA[], cudnnConvolutionMode_t *mode,
                                                          cudnnDataType_t *computeType) {
  ava_unsupported;
} /* convolution data type */

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                                const cudnnFilterDescriptor_t filterDesc, int nbDims,
                                                                int tensorOuputDimA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void *alpha,
                                                       const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                       const void *beta, const cudnnTensorDescriptor_t dbDesc,
                                                       void *db) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  ava_unsupported;
}

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(xDesc) ava_handle;
  ava_argument(dyDesc) ava_handle;
  ava_argument(convDesc) ava_handle;
  ava_argument(gradDesc) ava_handle;
  ava_argument(sizeInBytes) {
    ava_out;
    ava_buffer(1);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(cudnnHandle_t handle, const void *alpha,
                                                         const cudnnTensorDescriptor_t xDesc, const void *x,
                                                         const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                         const cudnnConvolutionDescriptor_t convDesc,
                                                         cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
                                                         size_t workSpaceSizeInBytes, const void *beta,
                                                         const cudnnFilterDescriptor_t dwDesc, void *dw) {
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }

  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(dyDesc) ava_handle;
  ava_argument(dy) ava_opaque;

  ava_argument(convDesc) ava_handle;
  ava_argument(workSpace) ava_opaque;

  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(dwDesc) ava_handle;
  ava_argument(dw) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t *algo) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  ava_unsupported;
}

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes) {
  ava_argument(wDesc) ava_handle;
  ava_argument(dyDesc) ava_handle;
  ava_argument(convDesc) ava_handle;
  ava_argument(dxDesc) ava_handle;
  ava_argument(sizeInBytes) {
    ava_out;
    ava_buffer(1);
  }
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(cudnnHandle_t handle, const void *alpha,
                                                       const cudnnFilterDescriptor_t wDesc, const void *w,
                                                       const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                       const cudnnConvolutionDescriptor_t convDesc,
                                                       cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
                                                       size_t workSpaceSizeInBytes, const void *beta,
                                                       const cudnnTensorDescriptor_t dxDesc, void *dx) {
  ava_async;
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(dx) ava_opaque;
  ava_argument(dxDesc) ava_handle;
  ava_argument(wDesc) ava_handle;
  ava_argument(w) ava_opaque;
  ava_argument(dyDesc) ava_handle;
  ava_argument(dy) ava_opaque;
  ava_argument(convDesc) ava_handle;
  ava_argument(workSpace) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
                                      const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
                                      void *colBuffer) {
  ava_unsupported;
}

/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                              cudnnSoftmaxMode_t mode, const void *alpha,
                                              const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                              const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                               cudnnSoftmaxMode_t mode, const void *alpha,
                                               const cudnnTensorDescriptor_t yDesc, const void *y,
                                               const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
                                               const cudnnTensorDescriptor_t dxDesc, void *dx) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
                                                      cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
                                                      int windowWidth, int verticalPadding, int horizontalPadding,
                                                      int verticalStride, int horizontalStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                                      cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
                                                      int *windowHeight, int *windowWidth, int *verticalPadding,
                                                      int *horizontalPadding, int *verticalStride,
                                                      int *horizontalStride) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
                                                      cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
                                                      int *nbDims, int windowDimA[], int paddingA[], int strideA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                                            const cudnnTensorDescriptor_t inputTensorDesc, int nbDims,
                                                            int outputTensorDimA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                                            const cudnnTensorDescriptor_t inputTensorDesc, int *n,
                                                            int *c, int *h, int *w) {
  ava_unsupported;
}

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                                       cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt,
                                                       double coef) {
  ava_unsupported;
} /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                                       cudnnActivationMode_t *mode, cudnnNanPropagation_t *reluNanOpt,
                                                       double *coef) {
  ava_unsupported;
} /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  ava_unsupported;
}

/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
                                                 const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                                 const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
                                                  const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
                                                  const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                  const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                                  const cudnnTensorDescriptor_t dxDesc, void *dx) {
  ava_unsupported;
}

/*
 * Create an instance of LRN (Local Response Normalization) descriptor
 * Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
 */
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) { ava_unsupported; }

/*
 * Uses a window [center-lookBehind, center+lookAhead], where
 * lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
 * Values of double parameters cast to tensor data type.
 */
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha,
                                                double lrnBeta, double lrnK) {
  ava_unsupported;
}
/*
 * Retrieve the settings currently stored in an LRN layer descriptor
 * Any of the provided pointers can be NULL (no corresponding value will be returned)
 */
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha,
                                                double *lrnBeta, double *lrnK) {
  ava_unsupported;
}

/* Destroy an instance of LRN descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) { ava_unsupported; }

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
                                                      cudnnLRNMode_t lrnMode, const void *alpha,
                                                      const cudnnTensorDescriptor_t xDesc, const void *x,
                                                      const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
                                                       cudnnLRNMode_t lrnMode, const void *alpha,
                                                       const cudnnTensorDescriptor_t yDesc, const void *y,
                                                       const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                       const cudnnTensorDescriptor_t xDesc, const void *x,
                                                       const void *beta, const cudnnTensorDescriptor_t dxDesc,
                                                       void *dx) {
  ava_unsupported;
}

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
    const void *x, const void *means,    /* if NULL, means are assumed to be zero */
    void *temp, void *temp2, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, means, dy, temp, temp2 */
    const void *x, const void *means,    /* if NULL, means are assumed to be zero */
    const void *dy, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
    void *dx,                                   /* output x differential */
    void *dMeans) {
  ava_unsupported;
} /* output means differential, can be NULL */

/*
 * Derives a tensor descriptor from layer data descriptor for BatchNormalization
 * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
 * bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                                        const cudnnTensorDescriptor_t xDesc,
                                                        cudnnBatchNormMode_t mode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
  ava_unsupported;
}

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc, const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,       /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
        (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
    const void *bnScale, const void *bnBias,

    /* MUST use factor=1 in the very first call of a complete training cycle.
       Use a factor=1/(1+n) at N-th call to the function to get
       Cumulative Moving Average (CMA) behavior
       CMA[n] = (x[1]+...+x[n])/n
       Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
       ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
       CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
    double exponentialAverageFactor,

    /* Used in Training phase only.
       runningMean = newMean*factor + runningMean*(1-factor) */
    void *resultRunningMean,
    /* Output in training mode, input in inference. Is the moving average
       of  variance[x] (factor is applied in the same way as for runningMean) */
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean, void *resultSaveInvVariance) {
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(yDesc) ava_handle;
  ava_argument(y) ava_opaque;
  ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
  ava_argument(bnScale) ava_opaque;
  ava_argument(bnBias) ava_opaque;
  ava_argument(resultRunningMean) ava_opaque;
  ava_argument(resultRunningVariance) ava_opaque;
  ava_argument(resultSaveMean) ava_opaque;
  ava_argument(resultSaveInvVariance) ava_opaque;
}

/* Computes y = relu(BN(x) + z). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData,

    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,

    double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean, void *resultSaveInvVariance,

    cudnnActivationDescriptor_t activationDesc, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(alpha) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }

  ava_argument(xDesc) ava_handle;
  ava_argument(xData) ava_opaque;
  ava_argument(yDesc) ava_handle;
  ava_argument(yData) ava_opaque;
  ava_argument(zDesc) ava_handle;
  ava_argument(zData) ava_opaque;

  ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
  ava_argument(bnScale) ava_opaque;
  ava_argument(bnBias) ava_opaque;

  ava_argument(resultRunningMean) ava_opaque;
  ava_argument(resultRunningVariance) ava_opaque;
  ava_argument(resultSaveMean) ava_opaque;
  ava_argument(resultSaveInvVariance) ava_opaque;

  ava_argument(activationDesc) ava_handle;
  ava_argument(workspace) ava_opaque;
  ava_argument(reserveSpace) ava_opaque;
}

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
 * bnScale gradient and bnBias gradient */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
    const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx,
    /* Shared tensor desc for the 4 tensors below */
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale, /* bnBias doesn't affect backpropagation */
    /* scale and bias diff are not backpropagated below this layer */
    void *dBnScaleResult, void *dBnBiasResult,
    /* Same epsilon as forward pass */
    double epsilon,

    /* Optionally cached intermediate results from
       forward pass */
    const void *savedMean, const void *savedInvVariance) {
  ava_argument(handle) ava_handle;
  ava_argument(alphaDataDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(betaDataDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(alphaParamDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(betaParamDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }

  ava_argument(xDesc) ava_handle;
  ava_argument(x) ava_opaque;
  ava_argument(dyDesc) ava_handle;
  ava_argument(dy) ava_opaque;
  ava_argument(dxDesc) ava_handle;
  ava_argument(dx) ava_opaque;
  ava_argument(dBnScaleBiasDesc) ava_handle;
  ava_argument(bnScale) ava_opaque;
  ava_argument(dBnScaleResult) ava_opaque;
  ava_argument(dBnBiasResult) ava_opaque;
  ava_argument(savedMean) ava_opaque;
  ava_argument(savedInvVariance) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,

    const void *alphaDataDiff, const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData, const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,

    /* Shared tensor desc for the 4 tensors below */
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScaleData,
    const void *bnBiasData,                                /* needed if there is activation */
    void *dBnScaleData, void *dBnBiasData, double epsilon, /* Same epsilon as forward pass */

    /* Optionally cached intermediate results from
       forward pass */
    const void *savedMean, const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  ava_argument(handle) ava_handle;
  ava_argument(alphaDataDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(betaDataDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(alphaParamDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }
  ava_argument(betaParamDiff) {
    ava_type_cast(const double *);
    ava_in;
    ava_buffer(1);
  }

  ava_argument(xDesc) ava_handle;
  ava_argument(xData) ava_opaque;
  ava_argument(yDesc) ava_handle;
  ava_argument(yData) ava_opaque;
  ava_argument(dxDesc) ava_handle;
  ava_argument(dxData) ava_opaque;
  ava_argument(dyDesc) ava_handle;
  ava_argument(dyData) ava_opaque;
  ava_argument(dzDesc) ava_handle;
  ava_argument(dzData) ava_opaque;

  ava_argument(dBnScaleBiasDesc) ava_handle;
  ava_argument(bnScaleData) ava_opaque;
  ava_argument(bnBiasData) ava_opaque;
  ava_argument(dBnScaleData) ava_opaque;
  ava_argument(dBnBiasData) ava_opaque;

  ava_argument(savedMean) ava_opaque;
  ava_argument(savedInvVariance) ava_opaque;

  ava_argument(activationDesc) ava_handle;
  ava_argument(workSpace) ava_opaque;
  ava_argument(reserveSpace) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                                                 cudnnSamplerType_t samplerType,
                                                                 cudnnDataType_t dataType, const int nbDims,
                                                                 const int dimA[]) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                                             const cudnnSpatialTransformerDescriptor_t stDesc,
                                                             const void *theta, void *grid) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                                              const cudnnSpatialTransformerDescriptor_t stDesc,
                                                              const void *dgrid, void *dtheta) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
                                                       const void *alpha, const cudnnTensorDescriptor_t xDesc,
                                                       const void *x, const void *grid, const void *beta,
                                                       cudnnTensorDescriptor_t yDesc, void *y) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
                                                        cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha,
                                                        const cudnnTensorDescriptor_t xDesc, const void *x,
                                                        const void *beta, const cudnnTensorDescriptor_t dxDesc,
                                                        void *dx, const void *alphaDgrid,
                                                        const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                        const void *grid, const void *betaDgrid, void *dgrid) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) { ava_unsupported; }

/*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes) { ava_unsupported; }

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                    float dropout, void *states, size_t stateSizeInBytes,
                                                    unsigned long long seed) {
  ava_unsupported;
}

/* Restores the dropout descriptor to a previously saved-off state */
cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                        float dropout, void *states, size_t stateSizeInBytes,
                                                        unsigned long long seed) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                    float *dropout, void **states, unsigned long long *seed) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
                                              const cudnnTensorDescriptor_t xdesc, const void *x,
                                              const cudnnTensorDescriptor_t ydesc, void *y, void *reserveSpace,
                                              size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
                                               const cudnnTensorDescriptor_t dydesc, const void *dy,
                                               const cudnnTensorDescriptor_t dxdesc, void *dx, void *reserveSpace,
                                               size_t reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode) {
  ava_unsupported;
}

/* Sequence data descriptor */

cudnnStatus_t CUDNNWINAPI cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
                                                    int nbDims, const int dimA[], const cudnnSeqDataAxis_t axes[],
                                                    size_t seqLengthArraySize, const int seqLengthArray[],
                                                    void *paddingFill) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                                                    cudnnDataType_t *dataType, int *nbDims, int nbDimsRequested,
                                                    int dimA[], cudnnSeqDataAxis_t axes[], size_t *seqLengthArraySize,
                                                    size_t seqLengthSizeRequested, int seqLengthArray[],
                                                    void *paddingFill) {
  ava_unsupported;
}

/* Multihead Attention */

/* Multi-head attention modes set in attention descriptor */

cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads,
                                                 double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec,
                                                 cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
                                                 cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize,
                                                 int vSize, int qProjSize, int kProjSize, int vProjSize, int oProjSize,
                                                 int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize,
                                                 int maxBeamSize) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads, double *smScaler, cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec, cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize, int *vSize, int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize, int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize, int *maxBeamSize) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
                                                       size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
                                                       size_t *reserveSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
                                                       cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
                                                       const void *weights, cudnnTensorDescriptor_t wDesc,
                                                       void **wAddr) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx, const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsQO[], const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc, const void *queries,
    const void *residuals, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries, const void *queries,
    const cudnnSeqDataDescriptor_t dkDesc, void *dkeys, const void *keys, const cudnnSeqDataDescriptor_t dvDesc,
    void *dvalues, const void *values, size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes,
    void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc, const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *weights, void *dweights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  ava_unsupported;
}

/*
 * CTC (Connectionist Temporal Classification) loss descriptor create/destory/set/get functions
 */
cudnnStatus_t CUDNNWINAPI cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
                                                      cudnnLossNormalizationMode_t normMode,
                                                      cudnnNanPropagation_t gradMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
                                                      cudnnLossNormalizationMode_t *normMode,
                                                      cudnnNanPropagation_t *gradMode) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) { ava_unsupported; }

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size)  */
    const void *probs,                       /* probabilities after softmax, in GPU memory */
    const int *labels,                       /* labels, in CPU memory */
    const int *labelLengths,                 /* the length of each label, in CPU memory */
    const int *inputLengths,                 /* the lengths of timing steps in each batch, in CPU memory */
    void *costs,                             /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
    const void *gradients,   /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes) {
  ava_unsupported;
} /* size of the workspace */

/* return the workspace size needed for ctc */
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size) */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the
                                                    dimensions are T,N,A. To compute costs
                                                    only, set it to NULL */
    const int *labels,                           /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    cudnnCTCLossAlgo_t algo,                     /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
  ava_unsupported;
} /* pointer to the returned workspace size */

cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc,
                                                      cudnnAlgorithm_t *algorithm) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src,
                                                       cudnnAlgorithmDescriptor_t dest) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                                                       cudnnAlgorithmDescriptor_t algoDesc, cudnnStatus_t status,
                                                       float time, size_t memory) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                                                       cudnnAlgorithmDescriptor_t *algoDesc, cudnnStatus_t *status,
                                                       float *time, size_t *memory) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
                                                     size_t *algoSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSaveAlgorithm(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, void *algoSpace,
                                             size_t algoSpaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(cudnnHandle_t handle, void *algoSpace, size_t algoSpaceSizeInBytes,
                                                cudnnAlgorithmDescriptor_t algoDesc) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack,
                                                            cudnnFusedOps_t ops) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,
                                                                  cudnnFusedOpsConstParamLabel_t paramLabel,
                                                                  const void *param) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                                                  cudnnFusedOpsConstParamLabel_t paramLabel,
                                                                  void *param, int *isNULL) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack,
                                                              cudnnFusedOps_t ops) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                                                    cudnnFusedOpsVariantParamLabel_t paramLabel,
                                                                    void *ptr) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                                                    cudnnFusedOpsVariantParamLabel_t paramLabel,
                                                                    void *ptr) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) { ava_unsupported; }

cudnnStatus_t CUDNNWINAPI cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                                                const cudnnFusedOpsConstParamPack_t constPack,
                                                size_t *workspaceSizeInBytes) {
  ava_unsupported;
}

cudnnStatus_t CUDNNWINAPI cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
                                               cudnnFusedOpsVariantParamPack_t varPack) {
  ava_unsupported;
}

/******** curand *********/
#include "cava/samples/cuda_common_spec/curand.h"
#include "cava/samples/cuda_common_spec/curand_unimplemented.h"

/******** cufft *********/
#include "cava/samples/cuda_common_spec/cufft_unimplemented.h"

/******* cusolver *********/
#include "cava/samples/cuda_common_spec/cusolver_unimplemented.h"

/******* cusparse *********/
#include "cava/samples/cuda_common_spec/cusparse/cusparse.h"
#include "cava/samples/cuda_common_spec/cusparse/cusparse_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/sparse_level1_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/sparse_level2_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/sparse_level3_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/preconditioner_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/sparse_level4_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/sparse_format_conversion_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/matrix_sorting_unimplemented.h"
#include "cava/samples/cuda_common_spec/cusparse/csr2csc_unimplemented.h"

/******* cudart *********/
#include "cava/samples/cuda_common_spec/cudart_unimplemented.h"

ava_begin_replacement;
EXPORTED __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void) { return cuda_last_error; }
ava_end_replacement;

__host__ __cudart_builtin__ const char *CUDARTAPI cudaGetErrorName(cudaError_t error) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) { ava_argument(stream) ava_handle; }

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) { ava_argument(stream) ava_handle; }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                                                        const void *func) {
  ava_disable_native_call;

  ava_argument(attr) {
    ava_out;
    ava_buffer(1);
  }

  ava_implicit_argument void *func_id = ava_metadata(func)->func_id;
  ava_argument(func_id) { ava_opaque; }
  ava_argument(func) { ava_opaque; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_func_get_attributes(
        attr, ((struct fatbin_function *)g_ptr_array_index(ava_metadata((void *)0)->fatbin_funcs, (intptr_t)func_id)),
        func_id);
    return ret;
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
  ava_disable_native_call;

  ava_argument(numBlocks) {
    ava_out;
    ava_buffer(1);
  }

  ava_implicit_argument void *func_id = ava_metadata(func)->func_id;
  ava_argument(func_id) { ava_opaque; }
  ava_argument(func) { ava_opaque; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_occupancy_max_active_blocks_per_multiprocessor(
        numBlocks,
        ((struct fatbin_function *)g_ptr_array_index(ava_metadata((void *)0)->fatbin_funcs, (intptr_t)func_id)),
        func_id, blockSize, dynamicSMemSize);
    return ret;
  }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
  ava_disable_native_call;

  ava_argument(numBlocks) {
    ava_out;
    ava_buffer(1);
  }

  ava_implicit_argument void *func_id = ava_metadata(func)->func_id;
  ava_argument(func_id) { ava_opaque; }
  ava_argument(func) { ava_opaque; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(
        numBlocks,
        ((struct fatbin_function *)g_ptr_array_index(ava_metadata((void *)0)->fatbin_funcs, (intptr_t)func_id)),
        func_id, blockSize, dynamicSMemSize, flags);
    return ret;
  }
}

/**
 * Initialization code in the generated code.
 */
ava_utility void __helper_guestlib_init_prologue() {
#ifdef AVA_PRELOAD_CUBIN
  /* Preload CUDA fat binaries */
  /* Read cubin number */
  int fd;
  ssize_t ret;
  int fatbin_num;
  fd = open("/cuda_dumps/fatbin-info.ava", O_RDONLY, 0666);
  if (fd == -1) {
    fprintf(stderr, "open /cuda_dumps/fatbin-info.ava [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = read(fd, (void *)&fatbin_num, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  AVA_DEBUG << "Fatbinary number = " << fatbin_num;
  int i;
  ava_metadata(NULL)->num_fatbins = 0;
  for (i = 0; i < fatbin_num; i++) {
    __helper_load_function_arg_info_guest();
  }
#endif
  guestlib_tf_opt_init();
}

ava_utility void __helper_guestlib_fini_prologue() { guestlib_tf_opt_fini(); }

ava_utility void __helper_worker_init_epilogue() {
#ifdef AVA_PRELOAD_CUBIN
  /* Preload CUDA fat binaries */
  fatbin_handle_list = g_ptr_array_new();
  /* Read cubin number */
  int fd;
  ssize_t ret;
  int fatbin_num;
  fd = open("/cuda_dumps/fatbin-info.ava", O_RDONLY, 0666);
  if (fd == -1) {
    fprintf(stderr, "open /cuda_dumps/fatbin-info.ava [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = read(fd, (void *)&fatbin_num, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  AVA_DEBUG << "Fatbinary number = " << fatbin_num;
  int i;
  void *fatCubin;
  void **fatbin_handle;
  for (i = 0; i < fatbin_num; i++) {
    fatCubin = malloc(sizeof(struct fatbin_wrapper));
    ret = read(fd, fatCubin, sizeof(struct fatbin_wrapper));
    if (ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    fatbin_handle = __helper_load_and_register_fatbin(fatCubin);
    g_ptr_array_add(fatbin_handle_list, (gpointer)fatbin_handle);
  }
  close(fd);
#endif
  worker_tf_opt_init();
}

ava_guestlib_init_prologue(__helper_guestlib_init_prologue());
ava_guestlib_fini_prologue(__helper_guestlib_fini_prologue());
ava_worker_init_epilogue(__helper_worker_init_epilogue());
