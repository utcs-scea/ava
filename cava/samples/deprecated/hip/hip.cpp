// clang-format off
ava_name("HIP");
ava_version("3.7.0");
ava_identifier(HIP);
ava_number(3);
ava_cxxflags(-D__HIP_PLATFORM_HCC__ -isystem /opt/rocm/include -I../headers -fpermissive);
ava_libs(-lamdhip64);
ava_export_qualifier();
// clang-format on

ava_begin_utility;
struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;
#include "hip_cpp_bridge.h"
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_runtime_api.h>
ava_end_utility;

hipError_t hipDeviceSynchronize(void) {}

hipError_t hipMalloc(void **dptr, size_t size) {
  ava_argument(dptr) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
}

hipError_t hipFree(void *ptr) {
  ava_argument(ptr) { ava_opaque; }
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes) {
  ava_argument(src) {
    ava_in;
    ava_buffer(sizeBytes);
  }
  ava_argument(dst) ava_opaque;
}

hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes) {
  ava_argument(src) ava_opaque;
  ava_argument(dst) {
    ava_out;
    ava_buffer(sizeBytes);
  }
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  ava_argument(src) ava_opaque;
  ava_argument(dst) ava_opaque;
}

hipError_t nw_hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
  ava_argument(dst) {
    ava_depends_on(kind);
    if (kind == hipMemcpyDeviceToHost) {
      ava_out;
      ava_buffer(sizeBytes);
    } else {
      ava_opaque;
    }
  }
  ava_argument(src) {
    ava_depends_on(kind);
    if (kind == hipMemcpyHostToDevice) {
      ava_in;
      ava_buffer(sizeBytes);
    } else {
      ava_opaque;
    }
  }
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes, hipStream_t stream) {
  ava_argument(src) {
    ava_in;
    ava_buffer(sizeBytes);
  }
  ava_argument(dst) ava_opaque;
  ava_argument(stream) ava_opaque;
}

hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  ava_argument(src) ava_opaque;
  ava_argument(dst) {
    ava_out;
    ava_buffer(sizeBytes);
  }
  ava_argument(stream) ava_opaque;
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  ava_argument(src) ava_opaque;
  ava_argument(dst) ava_opaque;
  ava_argument(stream) ava_opaque;
}

hipError_t nw_hipMemcpySync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
  ava_argument(dst) {
    ava_depends_on(kind);
    if (kind == hipMemcpyDeviceToHost) {
      ava_out;
      ava_buffer(sizeBytes);
    } else {
      ava_opaque;
    }
  }
  ava_argument(src) {
    ava_depends_on(kind);
    if (kind == hipMemcpyHostToDevice) {
      ava_in;
      ava_buffer(sizeBytes);
    } else {
      ava_opaque;
    }
  }
  ava_argument(stream) ava_opaque;
}

hipError_t hipGetDeviceCount(int *count) {
  ava_argument(count) {
    ava_out;
    ava_buffer(1);
  }
}

hipError_t nw_hipSetDevice(int deviceId) {}

hipError_t hipMemGetInfo(size_t *__free, size_t *total) {
  ava_argument(__free) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(total) {
    ava_out;
    ava_buffer(1);
  }
}

hipError_t nw_hipStreamCreate(hipStream_t *stream, hsa_agent_t *agent) {
  ava_argument(stream) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
  ava_argument(agent) {
    ava_out;
    ava_buffer(1);
  }
}

hipError_t nw_hipGetDevice(int *deviceId) {
  ava_argument(deviceId) {
    ava_out;
    ava_buffer(1);
  }
}

hipError_t hipInit(unsigned int flags) {}

hipError_t hipCtxGetCurrent(hipCtx_t *ctx) {
  ava_argument(ctx) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
}

hipError_t nw_hipStreamSynchronize(hipStream_t stream) {
  ava_argument(stream) { ava_opaque; }
}

hipError_t __do_c_hipGetDeviceProperties(char *prop, int deviceId) {
  ava_argument(prop) {
    ava_out;
    ava_buffer(sizeof(hipDeviceProp_t));
  }
}

ava_utility size_t hipLaunchKernel_extra_size(void **extra) {
  size_t size = 1;
  while (extra[size - 1] != HIP_LAUNCH_PARAM_END) size++;
  return size;
}

hipError_t __do_c_hipHccModuleLaunchKernel(hsa_kernel_dispatch_packet_t *aql, hipStream_t stream, void **kernelParams,
                                           char *extra, size_t extra_size, hipEvent_t start, hipEvent_t stop) {
  ava_argument(aql) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(kernelParams) {
    ava_in;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
  ava_argument(extra) {
    ava_in;
    ava_buffer(extra_size);
  }
  ava_argument(stream) { ava_opaque; }
  ava_argument(start) { ava_opaque; }
  ava_argument(stop) { ava_opaque; }
}

hipError_t __do_c_hipHccModuleLaunchMultiKernel(int numKernels, hsa_kernel_dispatch_packet_t *aql, hipStream_t stream,
                                                char *all_extra, size_t total_extra_size, size_t *extra_size,
                                                hipEvent_t *start, hipEvent_t *stop) {
  ava_argument(aql) {
    ava_in;
    ava_buffer(numKernels);
  }
  ava_argument(stream) { ava_opaque; }
  ava_argument(all_extra) {
    ava_in;
    ava_buffer(total_extra_size);
  }
  ava_argument(extra_size) {
    ava_in;
    ava_buffer(numKernels);
  }
  ava_argument(start) {
    ava_in;
    ava_buffer(numKernels);
    ava_element { ava_opaque; }
  }
  ava_argument(stop) {
    ava_in;
    ava_buffer(numKernels);
    ava_element { ava_opaque; }
  }
}

hsa_status_t HSA_API nw_hsa_system_major_extension_supported(uint16_t extension, uint16_t version_major,
                                                             uint16_t *version_minor, bool *result) {
  ava_argument(result) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(version_minor) {
    ava_in;
    ava_out;
    ava_buffer(1);
  }
}

hsa_status_t HSA_API nw_hsa_executable_create_alt(hsa_profile_t profile,
                                                  hsa_default_float_rounding_mode_t default_float_rounding_mode,
                                                  const char *options, hsa_executable_t *executable) {
  ava_argument(options) {
    ava_in;
    ava_buffer(strlen(options) + 1);
  }
  ava_argument(executable) {
    ava_out;
    ava_buffer(1);
  }
}

hsa_status_t HSA_API nw_hsa_isa_from_name(const char *name, hsa_isa_t *isa) {
  ava_argument(name) {
    ava_in;
    ava_buffer(strlen(name) + 1);
  }
  ava_argument(isa) {
    ava_out;
    ava_buffer(1);
  }
}

hipError_t hipPeekAtLastError(void) {}

hipError_t nw_hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId) {
  ava_argument(pi) {
    ava_out;
    ava_buffer(1);
  }
}

typedef uint16_t Elf_Half;
typedef uint32_t Elf_Word;
typedef int32_t Elf_Sword;
typedef uint64_t Elf_Xword;
typedef int64_t Elf_Sxword;

typedef uint32_t Elf32_Addr;
typedef uint32_t Elf32_Off;
typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;

#define Elf32_Half Elf_Half
#define Elf64_Half Elf_Half
#define Elf32_Word Elf_Word
#define Elf64_Word Elf_Word
#define Elf32_Sword Elf_Sword
#define Elf64_Sword Elf_Sword

#define EI_NIDENT 16

typedef struct Elf64_Ehdr {
  unsigned char e_ident[EI_NIDENT];
  Elf_Half e_type;
  Elf_Half e_machine;
  Elf_Word e_version;
  Elf64_Addr e_entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  Elf_Word e_flags;
  Elf_Half e_ehsize;
  Elf_Half e_phentsize;
  Elf_Half e_phnum;
  Elf_Half e_shentsize;
  Elf_Half e_shnum;
  Elf_Half e_shstrndx;
} Elf64_Ehdr;

ava_utility size_t calc_image_size(const void *image) {
  const Elf64_Ehdr *h = (Elf64_Ehdr *)image;

  return sizeof(Elf64_Ehdr) + h->e_shoff + h->e_shentsize * h->e_shnum;
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  ava_argument(module) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
  ava_argument(image) {
    ava_in;
    ava_buffer(calc_image_size(image));
  }
}

#include <stdint.h>

hsa_status_t HSA_API __do_c_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                                           hsa_executable_symbol_info_t attribute, char *value,
                                                           size_t max_value) {
  ava_argument(value) {
    ava_depends_on(max_value);
    ava_out;
    ava_buffer(max_value);
  }
}

hipError_t nw_hipCtxSetCurrent(hipCtx_t ctx) { ava_argument(ctx) ava_opaque; }

hipError_t hipEventCreate(hipEvent_t *event) {
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  ava_argument(event) ava_opaque;
  ava_argument(stream) ava_opaque;
}

hipError_t hipEventSynchronize(hipEvent_t event) { ava_argument(event) ava_opaque; }

hipError_t hipEventDestroy(hipEvent_t event) {
  ava_argument(event) { ava_opaque; }
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  ava_argument(ms) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(start) ava_opaque;
  ava_argument(stop) ava_opaque;
}

hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {
  ava_argument(module) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
  ava_argument(fname) {
    ava_in;
    ava_buffer(strlen(fname) + 1);
  }
}

hipError_t hipModuleUnload(hipModule_t module) {
  ava_argument(module) { ava_opaque; }
}

hipError_t nw_hipStreamDestroy(hipStream_t stream) {
  ava_argument(stream) { ava_opaque; }
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname) {
  ava_argument(function) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
  ava_argument(module) { ava_opaque; }
  ava_argument(kname) {
    ava_in;
    ava_buffer(strlen(kname) + 1);
  }
}

hipError_t hipGetLastError(void) {}

hipError_t hipMemset(void *dst, int value, size_t sizeBytes) { ava_argument(dst) ava_opaque; }

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  ava_argument(stream) ava_opaque;
  ava_argument(event) ava_opaque;
}

hsa_status_t HSA_API __do_c_hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void *value,
                                               size_t max_value) {
  ava_argument(value) {
    ava_depends_on(max_value);
    ava_out;
    ava_buffer(max_value);
  }
}

int __do_c_load_executable(const char *file_buf, size_t file_len, hsa_executable_t *executable, hsa_agent_t *agent) {
  ava_argument(file_buf) {
    ava_in;
    ava_buffer(file_len);
  }
  ava_argument(executable) {
    ava_in;
    ava_out;
    ava_buffer(1);
  }
  ava_argument(agent) {
    ava_in;
    ava_buffer(1);
  }
}

size_t __do_c_get_agents(hsa_agent_t *agents, size_t max_agents) {
  ava_argument(agents) {
    ava_out;
    ava_buffer(max_agents);
  }
}

size_t __do_c_get_isas(hsa_agent_t agents, hsa_isa_t *isas, size_t max_isas) {
  ava_argument(isas) {
    ava_out;
    ava_buffer(max_isas);
  }
}

size_t __do_c_get_kerenel_symbols(const hsa_executable_t *exec, const hsa_agent_t *agent,
                                  hsa_executable_symbol_t *symbols, size_t max_symbols) {
  ava_argument(exec) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(agent) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(symbols) {
    ava_out;
    ava_buffer(max_symbols);
  }
}

hsa_status_t HSA_API __do_c_query_host_address(uint64_t kernel_object_, char *kernel_header_) {
  ava_argument(kernel_object_) ava_opaque;
  ava_argument(kernel_header_) {
    ava_out;
    ava_buffer(sizeof(amd_kernel_code_t));
  }
}

hipError_t __do_c_get_kernel_descriptor(const hsa_executable_symbol_t *symbol, const char *name, hipFunction_t *f) {
  ava_argument(symbol) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(name) {
    ava_in;
    ava_buffer(strlen(name) + 1);
  }
  ava_argument(f) {
    ava_out;
    ava_buffer(1);
    ava_element { ava_opaque; }
  }
}

hipError_t nw_hipCtxGetDevice(hipDevice_t *device) {
  ava_argument(device) {
    ava_out;
    ava_buffer(1);
  }
}

hipError_t nw_lookup_kern_info(hipFunction_t f, struct nw_kern_info *info) {
  ava_argument(f) { ava_opaque; }
  ava_argument(info) {
    ava_out;
    ava_buffer(1);
  }
}
