#ifndef _AVA_GUESTLIB_NVVM_HELPER_H_
#define _AVA_GUESTLIB_NVVM_HELPER_H_
#include <nvvm.h>
#include <unistd.h>

void insert_compiled_result_size_map(nvvmProgram prog, size_t *bufferSizeRet);
void insert_program_log_size_map(nvvmProgram prog, size_t *bufferSizeRet);
size_t get_compiled_result_size_map(nvvmProgram prog);
size_t get_program_log_size_map(nvvmProgram prog);

#endif  // _AVA_GUESTLIB_NVVM_HELPER_H_
