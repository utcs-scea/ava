// clang-format off
ava_name("CUDA");
ava_version("10.1.0");
ava_identifier(CUDADRV);
ava_number(3);
ava_cflags(-I/usr/local/cuda-10.1/include);
ava_libs(-L/usr/local/cuda-10.1/lib64 -lcuda);
ava_soname(libcuda.so libcuda.so.1);
ava_export_qualifier();
// clang-format on

ava_non_transferable_types { ava_handle; }

ava_functions { ava_time_me; }

#include <cuda.h>

ava_begin_utility;
#include <time.h>
#include <stdio.h>
#include <sys/time.h>
#include <errno.h>
#include "common/logging.h"
ava_end_utility;

typedef struct {
  /* argument types */
  int func_argc;
  char func_arg_is_handle[64];
} Metadata;

ava_register_metadata(Metadata);

// ava_throughput_resource command_rate;
// ava_throughput_resource device_time;

CUresult CUDAAPI cuInit(unsigned int Flags);

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  ava_argument(device) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  ava_argument(pctx) {
    ava_out;
    ava_element(ava_allocates);
    ava_buffer(1);
  }
}

ava_utility size_t __helper_load_cubin_size(const char *fname) {
  FILE *fp;
  size_t cubin_size;

  fp = fopen(fname, "rb");
  if (!fp) {
    return 0;
  }
  fseek(fp, 0, SEEK_END);
  cubin_size = ftell(fp);
  fclose(fp);

  return cubin_size;
}

ava_utility void *__helper_load_cubin(const char *fname, size_t size) {
  FILE *fp;
  void *cubin;

  fp = fopen(fname, "rb");
  if (!fp) {
    return NULL;
  }
  cubin = malloc(size);
  size_t ret = fread(cubin, 1, size, fp);
  if (ret != size) {
    if (feof(fp)) {
      fprintf(stderr, "eof");
    } else if (ferror(fp)) {
      fprintf(stderr, "fread [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  fclose(fp);

  return cubin;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
  ava_disable_native_call;
  int res;

  ava_argument(module) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(fname) {
    ava_in;
    ava_buffer(strlen(fname) + 1);
  }

  ava_implicit_argument size_t size = __helper_load_cubin_size(fname);

  ava_implicit_argument void *cubin = __helper_load_cubin(fname, size);
  ava_argument(cubin) {
    ava_in;
    ava_buffer(size);
  }

  if (ava_is_worker) {
    if (!cubin) return CUDA_ERROR_FILE_NOT_FOUND;
    res = cuModuleLoadData(module, cubin);
    return res;
  }
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) { ava_async; }

ava_utility void ava_parse_function_args(const char *name, int *func_argc, char *func_arg_is_handle) {
  int i = 0, skip = 0;

  *func_argc = 0;
  if (strncmp(name, "_Z", 2)) abort();

  i = 2;
  while (i < strlen(name) && isdigit(name[i])) {
    skip = skip * 10 + name[i] - '0';
    i++;
  }

  i += skip;
  while (i < strlen(name)) {
    switch (name[i]) {
    case 'P':
      func_arg_is_handle[(*func_argc)++] = 1;
      if (i + 1 < strlen(name) && (name[i + 1] == 'f' || name[i + 1] == 'i'))
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

    case 'f':
    case 'i':
    case 'l':
      func_arg_is_handle[(*func_argc)++] = 0;
      break;

    case 'S':
      func_arg_is_handle[(*func_argc)++] = 1;
      while (i < strlen(name) && name[i] != '_') i++;
      break;

    case 'v':
      i = strlen(name);
      break;

    default:
      abort();
    }
    i++;
  }

  for (i = 0; i < *func_argc; i++) {
    ava_debug("function arg#%d it is %sa handle\n", i, func_arg_is_handle[i] ? "" : "not ");
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
  ava_parse_function_args(name, &ava_metadata(*hfunc)->func_argc, ava_metadata(*hfunc)->func_arg_is_handle);
}

ava_utility size_t cuLaunchKernel_extra_size(void **extra) {
  size_t size = 1;
  while (extra[size - 1] != CU_LAUNCH_PARAM_END) size++;
  return size;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  ava_argument(kernelParams) {
    ava_in;
    ava_buffer(ava_metadata(f)->func_argc);
    ava_element {
      if (ava_metadata(f)->func_arg_is_handle[ava_index]) {
        ava_type_cast(CUdeviceptr *);
        ava_buffer(sizeof(CUdeviceptr));
        // ava_handle;
      } else {
        ava_type_cast(int *);
        ava_buffer(sizeof(int));
      }
    }
  }
  ava_argument(extra) {
    ava_in;
    ava_buffer(cuLaunchKernel_extra_size(extra));
#warning The buffer size below states that every kernelParams[i] is 1 byte long.
    ava_element ava_buffer(1);
  }

  /*
  struct timeval start, end;
  uint64_t used_time = 0;
  if (ava_is_worker) {
      gettimeofday(&start, NULL);
  }
  */
  ava_execute();
  /*
  if (ava_is_worker) {
      // Set sync: export CUDA_LAUNCH_BLOCKING=1
      // cuCtxSynchronize();
      gettimeofday(&end, NULL);
      used_time = (end.tv_sec - start.tv_sec) * 1000000 +
          (end.tv_usec - start.tv_usec);
  }
  ava_consumes_resource(device_time, used_time);
  ava_consumes_resource(command_rate, 1);
  */
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  ava_async;
  ava_argument(ctx) ava_deallocates;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  ava_argument(dptr) {
    ava_out;
    ava_buffer(1);
    ava_element { /*ava_handle;*/
      ava_allocates;
    }
  }
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
  ava_argument(srcHost) {
    ava_in;
    ava_buffer(ByteCount);
  }
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  ava_argument(dstHost) {
    ava_out;
    ava_buffer(ByteCount);
  }
}

CUresult CUDAAPI cuCtxSynchronize(void);

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  ava_argument(driverVersion) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  ava_async;
  ava_argument(dptr) ava_deallocates;
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
  ava_argument(dptr) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(bytes) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(name) {
    ava_in;
    ava_buffer(strlen(name) + 1);
  }
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
  ava_argument(count) {
    ava_out;
    ava_buffer(1);
  }
}

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
  // https://devtalk.nvidia.com/default/topic/482869/cudagetexporttable-a-total-hack/
  ava_unsupported;
}
