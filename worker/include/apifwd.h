#ifndef __EXECUTOR_APIFWD_H__
#define __EXECUTOR_APIFWD_H__

#include "common/apis.h"
#include "common/devconf.h"
#include "common/ioctl.h"
#include "common/socket.h"
#include "common/task_queue.h"
#include "common/valconfig.h"
#include "dispatcher.h"
#include "object_list.h"

#ifdef __cplusplus
extern "C" {
#endif

extern struct exec_state ex_st;

#define OCL_LIB_FILE "/usr/local/cuda-10.0/lib64/libOpenCL.so.1.1"
//#define OCL_LIB_FILE "/opt/amdgpu-pro/lib/x86_64-linux-gnu/libOpenCL.so.1"
//#define OCL_LIB_FILE "/opt/intel/opencl/SDK/lib64/libOpenCL.so.1"

void load_ocl_lib(void);
int spawn_apifwd(void);
void *ocl_apifwd_handler(void *opaque);
void *tf_apifwd_handler(void *opaque);
void *cuda_apifwd_handler(void *opaque);
void *mvnc_apifwd_handler(void *opaque);

#define get_ptr_from_dstore(_vm_id, _ptr, _type)                                             \
  ((_ptr != NULL) ? (_type)((uintptr_t)ex_st.dstore.addr + VGPU_DSTORE_SIZE * (_vm_id - 1) + \
                            param->base.dstore_offset + (uintptr_t)_ptr)                     \
                  : NULL)

#ifdef __cplusplus
}
#endif

#endif
