#include "hip_cpp_bridge.h"

#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "check_env.h"
#include "hip/hcc_detail/code_object_bundle.hpp"
#include "hip_hcc_internal.h"
#include "nw/include/n_ava_channels.h"
#include "trace_helper.h"

using std::string;
struct ihipModuleSymbol_t {
  uint64_t _object{};  // The kernel object.
  amd_kernel_code_t const *_header{};
  string _name;  // TODO - review for performance cost.  Name is just used for debug.
};

namespace hip_impl {
inline std::uint64_t kernel_object(hsa_executable_symbol_t x) {
  std::uint64_t r = 0u;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &r);

  return r;
}
}  // namespace hip_impl

#define pinned_buf_size (1UL << 30)
static void *allocate_pinned_buf() {
  void *pinned;
  static int device = CHECK_ENV("NW_WORKER_HIP_DEVICE_ID", 0);
  assert(hipSetDevice(device) == hipSuccess);
  unsigned int flags = hipHostMallocPortable | hipHostMallocCoherent | hipHostMallocMapped;
  assert(hipHostMalloc(&pinned, pinned_buf_size, flags) == hipSuccess);
  return pinned;
}

void *pinned_bufs[N_AVA_CHANNELS] = {REP_N_CHANNELS(allocate_pinned_buf())};

extern "C" hipError_t nw_hipMemcpySync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind,
                                       hipStream_t stream) {
  void *pin_buf = pinned_bufs[get_ava_chan_no()];
  hipError_t e = hipSuccess;
  assert(sizeBytes < pinned_buf_size);
  switch (kind) {
  case hipMemcpyHostToDevice:
    memcpy(pin_buf, src, sizeBytes);
    e = hipMemcpyAsync(dst, pin_buf, sizeBytes, kind, stream);
    assert(e == hipSuccess);
    e = hipStreamSynchronize(stream);
    assert(e == hipSuccess);
    break;
  case hipMemcpyDeviceToHost:
    e = hipMemcpyAsync(pin_buf, src, sizeBytes, kind, stream);
    assert(e == hipSuccess);
    e = hipStreamSynchronize(stream);
    assert(e == hipSuccess);
    memcpy(dst, pin_buf, sizeBytes);
    break;
  case hipMemcpyDeviceToDevice:
    e = hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
    assert(e == hipSuccess);
    e = hipStreamSynchronize(stream);
    assert(e == hipSuccess);
    break;
  case hipMemcpyHostToHost:
  case hipMemcpyDefault:
  default:
    fprintf(stderr, "Invalid kind for nw_hipMemcpySync");
    assert("impossible");
  }
#if 0
    stream = ihipSyncAndResolveStream(stream);
    try {
        stream->locked_copySync(dst, src, sizeBytes, kind);
    } catch (ihipException& ex) {
        e = ex._code;
    }
#endif
  return e;
}

extern "C" hipError_t nw_hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src, int srcDevice,
                                            size_t sizeBytes, hipStream_t stream) {
  return hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream);
}

hipError_t nw_hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
  return hipMemcpy(dst, src, sizeBytes, kind);
}

extern "C" hipError_t nw_hipCtxGetDevice(hipDevice_t *device) { return hipCtxGetDevice(device); }

extern "C" hipError_t nw_hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId) {
  return hipDeviceGetAttribute(pi, attr, deviceId);
}

extern "C" hipError_t nw_hipStreamSynchronize(hipStream_t stream) { return hipStreamSynchronize(stream); }

extern "C" hipError_t nw_hipCtxSetCurrent(hipCtx_t ctx) { return hipCtxSetCurrent(ctx); }

extern "C" hipError_t nw_hipGetDevice(int *deviceId) { return hipGetDevice(deviceId); }

extern "C" hipError_t nw_hipSetDevice(int deviceId) { return hipSetDevice(deviceId); }

extern "C" hipError_t nw_hipStreamCreate(hipStream_t *stream, hsa_agent_t *agent) {
  hipError_t ret = hipStreamCreate(stream);
  if (!ret) {
    *agent = *static_cast<hsa_agent_t *>((*stream)->locked_getAv()->get_hsa_agent());
  }
  return ret;
}

extern "C" hipError_t nw_hipStreamDestroy(hipStream_t stream) { return hipStreamDestroy(stream); }

extern "C" hipError_t __do_c_hipGetDeviceProperties(char *prop, int deviceId) {
  return hipGetDeviceProperties((hipDeviceProp_t *)prop, deviceId);
}

hip_impl::Kernel_descriptor::Kernel_descriptor(std::uint64_t kernel_object, const std::string &name)
    : kernel_object_{kernel_object}, name_{name} {
  bool supported{false};
  std::uint16_t min_v{UINT16_MAX};
  auto r = hsa_system_major_extension_supported(HSA_EXTENSION_AMD_LOADER, 1, &min_v, &supported);

  if (r != HSA_STATUS_SUCCESS || !supported) return;

  hsa_ven_amd_loader_1_01_pfn_t tbl{};

  r = hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(tbl), reinterpret_cast<void *>(&tbl));

  if (r != HSA_STATUS_SUCCESS) return;
  if (!tbl.hsa_ven_amd_loader_query_host_address) return;

  r = tbl.hsa_ven_amd_loader_query_host_address(reinterpret_cast<void *>(kernel_object_),
                                                reinterpret_cast<const void **>(&kernel_header_));

  if (r != HSA_STATUS_SUCCESS) return;
}

extern "C" hsa_status_t HSA_API __do_c_query_host_address(uint64_t kernel_object_, char *kernel_header_) {
  amd_kernel_code_t const *kernel_ptr = nullptr;
  hsa_ven_amd_loader_1_01_pfn_t tbl{};

  auto r =
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(tbl), reinterpret_cast<void *>(&tbl));

  if (r != HSA_STATUS_SUCCESS) return r;
  if (!tbl.hsa_ven_amd_loader_query_host_address) return r;

  r = tbl.hsa_ven_amd_loader_query_host_address(reinterpret_cast<void *>(kernel_object_),
                                                reinterpret_cast<const void **>(&kernel_ptr));
  if (kernel_ptr) memcpy(kernel_header_, kernel_ptr, sizeof(*kernel_header_));
  return r;
}

extern "C" hipError_t __do_c_get_kernel_descriptor(const hsa_executable_symbol_t *symbol, const char *name,
                                                   hipFunction_t *f) {
  auto descriptor = new hip_impl::Kernel_descriptor(hip_impl::kernel_object(*symbol), std::string(name));
  *f = reinterpret_cast<hipFunction_t>(descriptor);
  return hipSuccess;
}

extern "C" hipError_t __do_c_mass_symbol_info(size_t n, const hsa_executable_symbol_t *syms, hsa_symbol_kind_t *types,
                                              hipFunction_t *descriptors, uint8_t *_agents, unsigned *offsets,
                                              char *pool, size_t pool_size) {
  hsa_agent_t *agents = (hsa_agent_t *)_agents;
  char *pool_cursor = pool, *pool_end = pool + pool_size;
  for (unsigned i = 0; i < n; i++) {
    hsa_executable_symbol_get_info(syms[i], HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &types[i]);
    hsa_executable_symbol_get_info(syms[i], HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &agents[i]);
    uint32_t name_sz = 0;
    hsa_executable_symbol_get_info(syms[i], HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_sz);
    if (name_sz > (pool_end - 1) - pool_cursor) {
      fprintf(stderr, "%s:%d: symbol name pool not large enough\n", __FILE__, __LINE__);
      abort();
    }

    hsa_executable_symbol_get_info(syms[i], HSA_EXECUTABLE_SYMBOL_INFO_NAME, pool_cursor);
    pool_cursor[name_sz] = '\0';

    if (types[i] == HSA_SYMBOL_KIND_KERNEL) {
      /* these descriptors live for the whole life of the program anyway
       * so just leak them for now
       */
      auto descriptor = new hip_impl::Kernel_descriptor(hip_impl::kernel_object(syms[i]), std::string(pool_cursor));
      descriptors[i] = reinterpret_cast<hipFunction_t>(descriptor);
    }

    offsets[i] = pool_cursor - pool;
    pool_cursor += name_sz + 1;
  }
  return hipSuccess;
}

extern "C" hsa_status_t HSA_API __do_c_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                                                      hsa_executable_symbol_info_t attribute,
                                                                      char *value, size_t max_value) {
  return hsa_executable_symbol_get_info(executable_symbol, attribute, (void *)value);
}

extern "C" hsa_status_t __do_c_hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void *value,
                                                  size_t max_value) {
  return hsa_agent_get_info(agent, attribute, value);
}

extern "C" int __do_c_load_executable(const char *file_buf, size_t file_len, hsa_executable_t *executable,
                                      hsa_agent_t *agent) {
  *executable = hip_impl::load_executable(string(file_buf, file_len), *executable, *agent);
  return 0;
}

extern "C" size_t __do_c_get_agents(hsa_agent_t *agents, size_t agents_len) {
  static const auto accelerators = hc::accelerator::get_all();
  size_t cur = 0;

  for (auto &&acc : accelerators) {
    auto agent = static_cast<hsa_agent_t *>(acc.get_hsa_agent());

    if (!agent || !acc.is_hsa_accelerator()) continue;
    if (cur < agents_len) agents[cur] = *agent;
    cur += 1;
  }
  assert(cur <= agents_len);
  return cur;
}

extern "C" size_t __do_c_get_isas(hsa_agent_t agent, hsa_isa_t *isas, size_t isas_len) {
  struct isa_arg {
    hsa_isa_t *isas;
    size_t cur;
    size_t isas_len;
  } arg = {isas, 0, isas_len};

  hsa_agent_iterate_isas(
      agent,
      [](hsa_isa_t x, void *__arg) {
        auto _arg = static_cast<isa_arg *>(__arg);
        if (_arg->cur < _arg->isas_len) _arg->isas[_arg->cur] = x;
        _arg->cur++;
        return HSA_STATUS_SUCCESS;
      },
      &arg);
  assert(arg.cur <= isas_len);
  return arg.cur;
}

extern "C" size_t __do_c_get_kerenel_symbols(const hsa_executable_t *exec, const hsa_agent_t *agent,
                                             hsa_executable_symbol_t *symbols, size_t symbols_len) {
  struct symbol_arg {
    hsa_executable_symbol_t *symbols;
    size_t cur;
    size_t symbols_len;
  } arg = {symbols, 0, symbols_len};

  auto copy_kernels = [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t s, void *__arg) {
    auto _arg = static_cast<symbol_arg *>(__arg);

    if (_arg->cur < _arg->symbols_len) _arg->symbols[_arg->cur] = s;
    _arg->cur++;
    return HSA_STATUS_SUCCESS;
  };
  hsa_executable_iterate_agent_symbols(*exec, *agent, copy_kernels, &arg);
  assert(arg.cur <= symbols_len);
  return arg.cur;
}

extern "C" hipError_t __do_c_hipHccModuleLaunchKernel(hsa_kernel_dispatch_packet_t *aql, hipStream_t stream,
                                                      void **kernelParams, char *_extra, size_t extra_size,
                                                      hipEvent_t start, hipEvent_t stop) {
  assert(kernelParams == nullptr);
  /*
    Kernel argument preparation.
  */
  stream = ihipSyncAndResolveStream(stream);

  hc::completion_future cf;
  auto av = &stream->lockopen_preKernelCommand()->_av;

  av->dispatch_hsa_kernel(aql, _extra, extra_size, (start || stop) ? &cf : nullptr, "");

  if (start) {
    start->attachToCompletionFuture(&cf, stream, hipEventTypeStartCommand);
  }
  if (stop) {
    stop->attachToCompletionFuture(&cf, stream, hipEventTypeStopCommand);
  }

  stream->lockclose_postKernelCommand("", av);

  return hipSuccess;
}

extern "C" hipError_t __do_c_hipHccModuleLaunchMultiKernel(int numKernels, hsa_kernel_dispatch_packet_t *aql,
                                                           hipStream_t stream, char *all_extra, size_t total_extra_size,
                                                           size_t *extra_size, hipEvent_t *start, hipEvent_t *stop) {
  // if (numKernels == 1) {
  //    fprintf(stderr, "Launching a batch of size 1\n");
  // }
  char *extra = all_extra;
  for (int i = 0; i < numKernels; i++) {
    void *new_extra[5] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, extra, HIP_LAUNCH_PARAM_BUFFER_SIZE, &extra_size[i],
                          HIP_LAUNCH_PARAM_END};
    hipError_t status =
        __do_c_hipHccModuleLaunchKernel(aql + i, stream, nullptr, extra, extra_size[i], start[i], stop[i]);
    if (status != hipSuccess) {
      fprintf(stderr, "Returning and error for a kernel!\n");
      return status;
    }
    extra += extra_size[i];
  }
  return hipSuccess;
}

hipError_t __do_c_hipHccModuleLaunchMultiKernel_and_memcpy(int numKernels, hsa_kernel_dispatch_packet_t *aql,
                                                           hipStream_t stream, char *all_extra, size_t total_extra_size,
                                                           size_t *extra_size, hipEvent_t *start, hipEvent_t *stop,
                                                           void *dst, const void *src, size_t sizeBytes,
                                                           hipMemcpyKind kind) {
  hipError_t ret;

  if (kind == hipMemcpyHostToDevice) {
    ret = nw_hipMemcpySync(dst, src, sizeBytes, kind, stream);
    if (ret != hipSuccess) return ret;
  }
  ret = __do_c_hipHccModuleLaunchMultiKernel(numKernels, aql, stream, all_extra, total_extra_size, extra_size, start,
                                             stop);
  if (ret != hipSuccess) return ret;
  if (kind == hipMemcpyDeviceToHost) ret = nw_hipMemcpySync(dst, src, sizeBytes, kind, stream);
  return ret;
}

extern "C" hsa_status_t HSA_API
nw_hsa_executable_create_alt(hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode,
                             const char *options, hsa_executable_t *executable) {
  return hsa_executable_create_alt(profile, default_float_rounding_mode, options, executable);
}

extern "C" hsa_status_t HSA_API nw_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                                                  hsa_executable_symbol_info_t attribute, void *value) {
  return hsa_executable_symbol_get_info(executable_symbol, attribute, value);
}

extern "C" hsa_status_t HSA_API nw_hsa_system_major_extension_supported(uint16_t extension, uint16_t version_major,
                                                                        uint16_t *version_minor, bool *result) {
  return hsa_system_major_extension_supported(extension, version_major, version_minor, result);
}

extern "C" hsa_status_t HSA_API nw_hsa_system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                                        size_t table_length, void *table) {
  return hsa_system_get_major_extension_table(extension, version_major, table_length, table);
}

extern "C" hsa_status_t HSA_API nw_hsa_isa_from_name(const char *name, hsa_isa_t *isa) {
  return hsa_isa_from_name(name, isa);
}

hipError_t nw_lookup_kern_info(hipFunction_t f, struct nw_kern_info *info) {
  info->_object = f->_object;
  info->workgroup_group_segment_byte_size = f->_header->workgroup_group_segment_byte_size;
  info->workitem_private_segment_byte_size = f->_header->workitem_private_segment_byte_size;
  return hipSuccess;
}
