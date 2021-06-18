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

ava_begin_replacement;
EXPORTED CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle,
                                                                       cublasPointerMode_t *mode) {
  /* XXX seems ok for tensorflow but might be wrong !FIXME */
  *mode = CUBLAS_POINTER_MODE_HOST;
  return CUBLAS_STATUS_SUCCESS;
}

EXPORTED CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle,
                                                                       cublasPointerMode_t mode) {
  /* XXX seems ok for tensorflow but might be wrong ! FIXME */
  assert(mode == CUBLAS_POINTER_MODE_HOST && "mode == CUBLAS_POINTER_MODE_HOST");
  return CUBLAS_STATUS_SUCCESS;
}
ava_end_replacement;

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

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

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

#include "cava/samples/cuda_common_spec/cudnn.h"
#include "cava/samples/cuda_common_spec/cudnn_unimplemented.h"

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
#include "cava/samples/cuda_common_spec/cudart.h"
#include "cava/samples/cuda_common_spec/cudart_unimplemented.h"

ava_begin_replacement;
EXPORTED __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void) { return cuda_last_error; }
ava_end_replacement;

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
