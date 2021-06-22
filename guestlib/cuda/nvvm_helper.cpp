#include <absl/container/flat_hash_map.h>
#include <fmt/core.h>
#include <nvvm.h>

#include "common/logging.h"

absl::flat_hash_map<nvvmProgram, size_t> compiled_result_size_map = {};
absl::flat_hash_map<nvvmProgram, size_t> program_log_size_map = {};

void insert_compiled_result_size_map(nvvmProgram prog, size_t *bufferSizeRet) {
  compiled_result_size_map.insert({prog, *bufferSizeRet});
}

void insert_program_log_size_map(nvvmProgram prog, size_t *bufferSizeRet) {
  program_log_size_map.insert({prog, *bufferSizeRet});
}

size_t get_compiled_result_size_map(nvvmProgram prog) {
  auto result = compiled_result_size_map.find(prog);
  if (result != compiled_result_size_map.end()) {
    return result->second;
  } else {
    AVA_FATAL << "Expect nvvmGetCompiledResultSize to be called first";
  }
}

size_t get_program_log_size_map(nvvmProgram prog) {
  auto result = program_log_size_map.find(prog);
  if (result != program_log_size_map.end()) {
    return result->second;
  } else {
    AVA_FATAL << "Expected nvvmGetProgramLogSize to be called first";
  }
}
