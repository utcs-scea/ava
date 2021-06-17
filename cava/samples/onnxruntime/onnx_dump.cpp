// clang-format off
ava_name("CUDA Runtime for ONNX");
ava_version("10.1.0");
ava_identifier(ONNX_DUMP);
ava_number(10);
ava_cxxflags(-I/usr/local/cuda-10.1/include -I${CMAKE_SOURCE_DIR}/cava/headers);
ava_libs(-L/usr/local/cuda-10.1/lib64 -lcudart -lcuda -lcublas -lcudnn -lcufft -lcurand -lcusparse -lcusolver);
ava_guestlib_srcs(../common/extensions/cudart_10.1_utilities.cpp);
ava_worker_srcs(../common/extensions/cudart_10.1_utilities.cpp);
ava_export_qualifier();
ava_soname(libcuda.so libcuda.so.1 libcudart.so.10 libcudart.so.10.1 libcublas.so.10 libcublasLt.so.10 libcudnn.so.7 libcufft.so.10 libcurand.so.10 libcusolver.so.10 libcusparse.so.10);
// clang-format on

/**
 * The spec is used to dump the fat binaries and CUDA functions from
 * ONNXruntime library.
 * Compile by
 * ./nwcc samples/onnxruntime/onnx_dump.cpp -I /usr/local/cuda-10.1/include -I headers `pkg-config --cflags glib-2.0`
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
#include <errno.h>
#include <stdio.h>
#include <glib.h>
#include <algorithm>

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
ava_end_utility;

ava_type(cudaError_t) { ava_success(cudaSuccess); }

ava_type(cublasStatus_t) { ava_success(CUBLAS_STATUS_SUCCESS); }

ava_type(cudnnStatus_t) { ava_success(CUDNN_STATUS_SUCCESS); }

ava_type(CUresult) { ava_success(CUDA_SUCCESS); }

ava_type(curandStatus_t) { ava_success(CURAND_STATUS_SUCCESS); }

ava_type(cufftResult) { ava_success(CUFFT_SUCCESS); }

ava_type(cusparseStatus_t) { ava_success(CUSPARSE_STATUS_SUCCESS); }

ava_type(cusolverStatus_t) { ava_success(CUSOLVER_STATUS_SUCCESS); }
typedef struct {
  int num_fatbins;
  int fd_functions;

  /* argument types */
  GHashTable *fatbin_funcs; /* for NULL, the hash table */
  int num_funcs;
  struct fatbin_function *func; /* for functions */

  /* global states */
  CUmodule cur_module;
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
    ava_type_cast(void *);
    ava_in;
    ava_buffer(((struct fatBinaryHeader *)ava_self->ptr)->headerSize +
               ((struct fatBinaryHeader *)ava_self->ptr)->fatSize);
    ava_lifetime_static;
  }
  ava_field(data_ptr) { ava_self->data_ptr = 0; }
}

ava_type(struct cudaDeviceProp);

ava_type(struct cudaPointerAttributes) {
  ava_field(devicePointer) ava_handle;
  ava_field(hostPointer) ava_opaque;
};

/* APIs needed for a minimal program */

char CUDARTAPI __cudaInitModule(void **fatCubinHandle) {
  ava_argument(fatCubinHandle) {
    ava_in;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

ava_utility void __helper_dump_fatbin(void *fatCubin, GHashTable **fatbin_funcs, int *num_funcs) {
  struct fatbin_wrapper *wp = static_cast<struct fatbin_wrapper *>(fatCubin);
  struct fatBinaryHeader *fbh = reinterpret_cast<struct fatBinaryHeader *>(wp->ptr);
  int fd, ret;

  /* Increase fatbin counter */
  static int fatbin_num = 0;
  fatbin_num++;
  if (ava_is_worker) {
    char *file_name = "/tmp/fatbin-info.ava";
    fd = open(file_name, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
      fprintf(stderr, "open %s [errno=%d, errstr=%s] at %s:%d", file_name, errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    AVA_DEBUG << "Fatbinary counter = " << fatbin_num;
    ret = write(fd, (const void *)&fatbin_num, sizeof(int));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    ret = lseek(fd, 0, SEEK_END);
    if (ret == -1) {
      fprintf(stderr, "lseek [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    ret = write(fd, (const void *)wp, sizeof(struct fatbin_wrapper));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    close(fd);
  }

  /* Dump fat binary to a file */
  char fatbin_filename[32];
  if (ava_is_worker) {
    sprintf(fatbin_filename, "/tmp/fatbin-%d.ava", ava_metadata(NULL)->num_fatbins);
    fd = open(fatbin_filename, O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (fd == -1) {
      fprintf(stderr, "open %s [errno=%d, errstr=%s] at %s:%d", fatbin_filename, errno, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    AVA_DEBUG << "Dump fatbinary to " << fatbin_filename;
    ret = write(fd, (const void *)wp->ptr, fbh->headerSize + fbh->fatSize);
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    close(fd);
  }

  /* Execute cuobjdump and construct function information table */
  FILE *fp_pipe;
  char line[2048];
  int i, ordinal;
  size_t size;
  char name[MAX_KERNEL_NAME_LEN]; /* mangled name */
  struct fatbin_function *func;

  /* Create the hash table */
  if (*fatbin_funcs == NULL) {
    *fatbin_funcs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
    *num_funcs = 0;
  }

  /* Add separator to the functions of different fatbinaries */
  if (ava_is_worker) {
    if (ava_metadata(NULL)->fd_functions != 0) {
      size = 0;
      ret = write(ava_metadata(NULL)->fd_functions, (const void *)&size, sizeof(size_t));
      if (ret == -1) {
        fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    }
  }

  /*  Open the command pipe for reading */
  char pip_command[80];
  sprintf(pip_command, "/usr/local/cuda-10.1/bin/cuobjdump -elf /tmp/fatbin-%d.ava", ava_metadata(NULL)->num_fatbins);
  fp_pipe = popen(pip_command, "r");
  assert(fp_pipe);

  /* Open function argument dump file */
  int function_arg_fd;
  char function_arg_filename[32];
  if (ava_is_worker) {
    sprintf(function_arg_filename, "/tmp/function_arg-%d.ava", ava_metadata(NULL)->num_fatbins);
    function_arg_fd = open(function_arg_filename, O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (function_arg_fd == -1) {
      fprintf(stderr, "open %s [errno=%d, errstr=%s] at %s:%d", function_arg_filename, errno, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    AVA_DEBUG << "Dump function argument info to " << function_arg_filename;
  }

  while (fgets(line, sizeof(line), fp_pipe) != NULL) {
    /* Search functions */
    if (strncmp(line, ".nv.info._Z", 11) == 0) {
      sprintf(name, line + 9, strlen(line) - 10);
      assert(strlen(line) - 10 < MAX_KERNEL_NAME_LEN);
      name[strlen(line) - 10] = '\0';
      ava_debug("[%d] %s@", *num_funcs, name);

      /* Create a new hash table entry */
      func = (struct fatbin_function *)g_malloc(sizeof(struct fatbin_function));
      memset(func, 0, sizeof(struct fatbin_function));

      // TODO: parse function name to determine whether the
      // arguments are handles

      /* Search parameters */
      func->argc = 0;
      char *fgets_ret;
      while (fgets(line, sizeof(line), fp_pipe) != NULL) {
        i = 0;
        while (i < strlen(line) && isspace(line[i])) i++;
        /* Empty line means reaching the end of the function info */
        if (i == strlen(line)) break;

        if (strncmp(&line[i], "Attribute:", 10) == 0) {
          i += 10;
          while (i < strlen(line) && isspace(line[i])) i++;
          if (strncmp(&line[i], "EIATTR_KPARAM_INFO", 18) == 0) {
            /* Skip the format line */
            fgets_ret = fgets(line, sizeof(line), fp_pipe);
            if (fgets_ret == NULL) {
              if (feof(fp_pipe)) {
                fprintf(stderr, "End of file");
              } else if (ferror(fp_pipe)) {
                fprintf(stderr, "fgets [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
                exit(EXIT_FAILURE);
              }
            }
            fgets_ret = fgets(line, sizeof(line), fp_pipe);
            if (fgets_ret == NULL) {
              if (feof(fp_pipe)) {
                fprintf(stderr, "End of file");
              } else if (ferror(fp_pipe)) {
                fprintf(stderr, "fgets [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
                exit(EXIT_FAILURE);
              }
            }

            /* Get ordinal and size */
            i = 0;
            while (i < strlen(line) && line[i] != 'O') i++;
            sscanf(&line[i], "Ordinal\t: 0x%x", &ordinal);
            while (i < strlen(line) && line[i] != 'S') i++;
            sscanf(&line[i], "Size\t: 0x%lx", &size);

            i = func->argc;
            AVA_DEBUG << "ordinal=" << ordinal << ", size=" << size;
            assert(ordinal < MAX_KERNEL_ARG);
            func->args[ordinal].size = size;
            ++(func->argc);
          }
        }
      }

      ++(*num_funcs);

      /* Dump the function argument sizes to file */
      if (ava_is_worker) {
        size = strlen(name) + 1;
        ret = write(function_arg_fd, (void *)&size, sizeof(size_t));
        if (ret == -1) {
          fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
          exit(EXIT_FAILURE);
        }
        ret = write(function_arg_fd, (void *)name, size);
        if (ret == -1) {
          fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
          exit(EXIT_FAILURE);
        }
        ret = write(function_arg_fd, (void *)func, sizeof(struct fatbin_function));
        if (ret == -1) {
          fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
          exit(EXIT_FAILURE);
        }
      }

      /* Insert the function into hash table */
      if (g_hash_table_lookup(*fatbin_funcs, name) != NULL)
        g_free(func);
      else
        g_hash_table_insert((*fatbin_funcs), g_strdup(name), (gpointer)func);
      // func = (struct fatbin_function *)g_hash_table_lookup(*fatbin_funcs, name);
    }
  }

  if (ava_is_worker) close(function_arg_fd);

  pclose(fp_pipe);
  ++(ava_metadata(NULL)->num_fatbins);
}

ava_utility void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle) {
  int ret;
  if (ava_metadata(NULL)->cuinit_called == 0) {
    ret = cuInit(0);
    AVA_DEBUG << "cuInit in " << __func__ << " ret=" << ret;
    assert(ret == CUDA_SUCCESS && "CUDA driver init failed");
    ava_metadata(NULL)->cuinit_called = 1;
  }
  __cudaInitModule(handle);
  ava_metadata(NULL)->cur_module = NULL;
  ret = cuModuleLoadData(&ava_metadata(NULL)->cur_module, (void *)fatCubin->ptr);
  (void)ret;
  assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  ava_argument(fatCubin) {
    ava_type_cast(struct fatbin_wrapper *);
    ava_in;
    ava_buffer(1);
    ava_lifetime_static;
  }

  void **ret = reinterpret_cast<void **>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(__helper_cubin_num(ret) + 1);
    ava_element {
      if (ret[ava_index] != NULL) ava_handle;
    }
    ava_allocates;
    ava_lifetime_manual;
  }

  __helper_dump_fatbin(fatCubin, &ava_metadata(NULL)->fatbin_funcs, &ava_metadata(NULL)->num_funcs);

  if (ava_is_worker) {
    //__helper_print_fatcubin_info(fatCubin, ret);
    __helper_init_module((struct fatbin_wrapper *)fatCubin, ret);
  }
}

void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
  ava_disable_native_call;

  ava_argument(fatCubinHandle) {
    ava_in;
    ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
    ava_element {
      if (fatCubinHandle[ava_index] != NULL) ava_handle;
    }
    ava_deallocates;
  }

  if (ava_is_worker) {
    __helper_unregister_fatbin(fatCubinHandle);
  }
}

ava_utility void __helper_dump_cuda_function(char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
                                             uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
  int fd = ava_metadata(NULL)->fd_functions;
  if (fd == 0) {
    fd = open("/tmp/fatfunction.ava", O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (fd == -1) {
      fprintf(stderr, "open /tmp/fatfunction.ava [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    ava_metadata(NULL)->fd_functions = fd;
  }

  size_t size;
  int exists;
  ssize_t ret;
  size = strlen(deviceFun) + 1;
  ret = write(fd, (const void *)&size, sizeof(size_t));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = write(fd, (const void *)deviceFun, size);
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  size = strlen(deviceName) + 1;
  ret = write(fd, (const void *)&size, sizeof(size_t));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = write(fd, (const void *)deviceName, size);
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = write(fd, (const void *)&thread_limit, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  exists = (tid != NULL);
  ret = write(fd, (const void *)&exists, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  if (exists) {
    ret = write(fd, (const void *)tid, sizeof(uint3));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  exists = (bid != NULL);
  ret = write(fd, (const void *)&exists, sizeof(int));
  if (exists) {
    ret = write(fd, (const void *)bid, sizeof(uint3));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  exists = (bDim != NULL);
  ret = write(fd, (const void *)&exists, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  if (exists) {
    ret = write(fd, (const void *)bDim, sizeof(dim3));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  exists = (gDim != NULL);
  ret = write(fd, (const void *)&exists, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  if (exists) {
    ret = write(fd, (const void *)gDim, sizeof(dim3));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  exists = (wSize != NULL);
  ret = write(fd, (const void *)&exists, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  if (exists) {
    ret = write(fd, (const void *)wSize, sizeof(int));
    if (ret == -1) {
      fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  }
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim, int *wSize) {
  ava_disable_native_call;

  if (ava_is_worker) __helper_dump_cuda_function(deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

  ava_debug(
      "Register hostFun=%p, deviceFun=%s, deviceName=%s, thread_limit=%d, tid={%d,%d,%d}, bid={%d,%d,%d}, "
      "bDim={%d,%d,%d}, gDim={%d,%d,%d}",
      (void *)hostFun, deviceFun, deviceName, thread_limit, tid ? tid->x : 0, tid ? tid->y : 0, tid ? tid->z : 0,
      bid ? bid->x : 0, bid ? bid->y : 0, bid ? bid->z : 0, bDim ? bDim->x : 0, bDim ? bDim->y : 0, bDim ? bDim->z : 0,
      gDim ? gDim->x : 0, gDim ? gDim->y : 0, gDim ? gDim->z : 0);

  ava_argument(fatCubinHandle) {
    ava_in;
    ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
    ava_element {
      if (fatCubinHandle[ava_index] != NULL) ava_handle;
    }
  }

  ava_argument(hostFun) { ava_opaque; }

  ava_argument(deviceFun) {
    ava_in;
    ava_buffer(strlen(deviceFun) + 1);
  }

  ava_argument(deviceName) {
    ava_in;
    ava_buffer(strlen(deviceName) + 1);
  }

  __helper_assosiate_function_dump(ava_metadata(NULL)->fatbin_funcs, &ava_metadata(hostFun)->func, (void *)hostFun,
                                   deviceName);

  ava_argument(tid) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(bid) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(bDim) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(gDim) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(wSize) {
    ava_in;
    ava_buffer(1);
  }

  if (ava_is_worker) {
    __helper_register_function(ava_metadata(hostFun)->func, hostFun, ava_metadata(NULL)->cur_module, deviceName);
  }
}

ava_begin_replacement;
EXPORTED void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                                          const char *deviceName, int ext, size_t size, int constant, int global) {}

EXPORTED void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
#warning This API is called for CUDA 10.1 and 10.2, but it seems to be able to be ignored.
}
ava_end_replacement;

__host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                            size_t sharedMem,  // CHECKME: default argument in header
                            void *stream) {
  ava_argument(stream) { ava_handle; }
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
  ava_argument(gridDim) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(blockDim) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(sharedMem) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(stream) {
    ava_type_cast(CUstream *);
    ava_out;
    ava_buffer(1);
    ava_element { ava_handle; }
  }
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                                                size_t sharedMem, cudaStream_t stream) {
  ava_disable_native_call;

  ava_argument(func) { ava_opaque; }

  ava_argument(args) {
    ava_in;
    ava_buffer(ava_metadata(func)->func->argc);
    ava_element {
      // FIXME: use the generated index name in the spec to
      // reference the outer loop's loop index at this moment.
      if (ava_metadata(func)->func->args[__args_index_0].is_handle) {
        ava_type_cast(void *);
        ava_buffer(ava_metadata(func)->func->args[__args_index_0].size);
        // ava_element ava_handle;
      } else {
        ava_type_cast(void *);
        ava_buffer(ava_metadata(func)->func->args[__args_index_0].size);
      }
    }
  }

  ava_argument(stream) { ava_handle; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_launch_kernel(ava_metadata(func)->func, func, gridDim, blockDim, args, sharedMem, stream);
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

  ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count) { ava_argument(devPtr) ava_opaque; }

__host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) {
  ava_argument(attributes) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(ptr) {
    // ava_type_cast(CUdeviceptr);
    // ava_handle;
    ava_opaque;
  }
}

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
  return cudaSuccess;
}
ava_end_replacement;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void);

__host__ __cudart_builtin__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}

/* CUDA driver API */

CUresult CUDAAPI cuInit(unsigned int Flags);

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
  }
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  ava_argument(device) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  ava_argument(name) {
    ava_out;
    ava_buffer(len);
  }
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  ava_argument(uuid) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  ava_argument(pi) {
    ava_out;
    ava_buffer(1);
  }
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
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  ava_argument(pctx) {
    ava_out;
    ava_element(ava_allocates);
    ava_buffer(1);
  }
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

CUresult CUDAAPI cuStreamQuery(CUstream hStream) { ava_argument(hStream) ava_handle; }

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) { ava_argument(hStream) ava_handle; }

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  ava_argument(dptr) {
    ava_out;
    ava_buffer(1);
    ava_element {
      ava_opaque;
      ava_allocates;
    }
  }
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
  ava_argument(pp) {
    ava_out;
    ava_buffer(1);
    ava_element {
      ava_buffer(bytesize);
      ava_buffer_allocator(__helper_cu_mem_host_alloc_portable, __helper_cu_mem_host_free);
      ava_lifetime_manual;
      ava_allocates;
      ava_no_copy;
    }
  }

  ava_execute();
  ava_metadata(*pp)->is_pinned = 1;
}

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
  ava_argument(dstDevice) ava_opaque;

  ava_argument(srcHost) {
    ava_in;
    ava_buffer(ByteCount);
    if (ava_metadata(srcHost)->is_pinned) {
      ava_lifetime_manual;
    } else {
      ava_lifetime_manual;
    }
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
    if (ava_metadata(dstHost)->is_pinned) {
      ava_lifetime_manual;
    } else {
      ava_lifetime_manual;
    }
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

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) { ava_argument(dstDevice) ava_opaque; }

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) { ava_argument(dstDevice) ava_opaque; }

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

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  ava_argument(bytes) {
    ava_out;
    ava_buffer(1);
  }
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
}

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  ava_argument(phEvent) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) { ava_argument(hEvent) ava_handle; }

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  ava_argument(hEvent) ava_handle;
  ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) { ava_argument(hEvent) ava_handle; }

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
  ava_argument(pMilliseconds) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(hStart) ava_handle;
  ava_argument(hEnd) ava_handle;
}

CUresult cuEventDestroy(CUevent hEvent) { ava_argument(hEvent) ava_handle; }

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  ava_argument(hStream) ava_handle;
  ava_argument(hEvent) ava_handle;

  /*
#warning Fix the update of the buffers that are copied asynchronously.
  struct async_buffer_list *async_buffers;
  async_buffers = __helper_load_async_buffer_list(
          &ava_metadata(hStream)->async_buffers);

  ava_implicit_argument
  int num_buffers = async_buffers == NULL ? 0 : async_buffers->num_buffers;

  ava_implicit_argument
  size_t *buffer_sizes = async_buffers == NULL ? NULL : async_buffers->buffer_sizes;
  ava_argument(buffer_sizes) {
      ava_in; ava_buffer(num_buffers);
  }

  ava_implicit_argument
  void **buffers = async_buffers == NULL ? NULL : async_buffers->buffers;
  ava_argument(buffers) {
      ava_in; ava_buffer(num_buffers);
      ava_element {
          ava_out;
          ava_buffer(buffer_sizes[ava_index]);
      }
  }

  if (async_buffers != NULL)
      free(async_buffers);
  */
}

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

#include "cava/samples/cuda_common_spec/cudadrv_unimplemented.h"

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
  ava_argument(A) ava_opaque;
  ava_argument(B) ava_opaque;
  ava_argument(C) ava_opaque;
  /* XXX I _think_ these are always device pointers for tensorflow ! */
  ava_argument(alpha) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(beta) {
    ava_in;
    ava_buffer(1);
  }
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

cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  ava_argument(convDesc) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  ava_argument(filterDesc) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  ava_argument(poolingDesc) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  ava_argument(tensorDesc) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  ava_argument(filterDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  ava_argument(poolingDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  ava_argument(tensorDesc) ava_handle;
}

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
  ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
  ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                          int arrayLength, /* nbDims-2 size */
                                                          const int padA[], const int filterStrideA[],
                                                          const int dilationA[], cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t computeType) /* convolution data type */
{
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
  ava_argument(handle) ava_handle;
  ava_argument(streamId) ava_handle;
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
                                                     int nbDims, const int dimA[], const int strideA[]) {
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

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

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

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                                                        const void *func) {
  ava_disable_native_call;

  ava_argument(attr) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(func) { ava_opaque; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_func_get_attributes(attr, ava_metadata(func)->func, func);
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

  ava_argument(func) { ava_opaque; }
  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_occupancy_max_active_blocks_per_multiprocessor(numBlocks, ava_metadata(func)->func, func, blockSize,
                                                                  dynamicSMemSize);
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

  ava_argument(func) { ava_opaque; }

  cudaError_t ret;
  if (ava_is_worker) {
    ret = __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(numBlocks, ava_metadata(func)->func, func,
                                                                             blockSize, dynamicSMemSize, flags);
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

/* ONNX */

const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}
