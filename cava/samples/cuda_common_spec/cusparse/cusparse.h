#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_H_
#include <cusparse.h>

//##############################################################################
//# HELPER ROUTINES
//##############################################################################

const char *CUSPARSEAPI cusparseGetErrorName(cusparseStatus_t status) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}

const char *CUSPARSEAPI cusparseGetErrorString(cusparseStatus_t status) {
  const char *ret = reinterpret_cast<const char *>(ava_execute());
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_static;
  }
}

#endif // _AVA_SAMPLES_CUDA_COMMON_SPEC_CUSPARSE_H_
