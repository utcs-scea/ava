#include <absl/container/flat_hash_map.h>
#include <nvvm.h>

#include <iostream>

#include "guestlib/guest_config.h"
#include "guestlib/guest_context.h"

void insert_compiled_result_size_map(nvvmProgram prog, size_t *bufferSizeRet) {
  auto gctx = ava::GuestContext::instance();
  gctx->compiled_result_size_map.insert({prog, *bufferSizeRet});
}

void insert_program_log_size_map(nvvmProgram prog, size_t *bufferSizeRet) {
  auto gctx = ava::GuestContext::instance();
  gctx->program_log_size_map.insert({prog, *bufferSizeRet});
}

size_t get_compiled_result_size_map(nvvmProgram prog) {
  auto gctx = ava::GuestContext::instance();
  auto result = gctx->compiled_result_size_map.find(prog);
  if (result != gctx->compiled_result_size_map.end()) {
    return result->second;
  } else {
    std::cerr << fmt::format("need to call nvvmGetCompiledResultSize first\n");
    abort();
  }
}

size_t get_program_log_size_map(nvvmProgram prog) {
  auto gctx = ava::GuestContext::instance();
  auto result = gctx->program_log_size_map.find(prog);
  if (result != gctx->program_log_size_map.end()) {
    return result->second;
  } else {
    std::cerr << fmt::format("need to call nvvmGetProgramLogSize first\n");
    abort();
  }
}
