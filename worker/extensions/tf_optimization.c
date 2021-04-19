#include "common/endpoint_lib.h"
#include "common/extensions/tf_optimization.h"

#include <glib.h>
#include <stdint.h>

#ifdef AVA_PRELOAD_CUBIN
GPtrArray *fatbin_handle_list;
#endif
GTree *gpu_address_set; /* Not used but referenced in utility function */


// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
void guestlib_tf_opt_init(void) {}
void guestlib_tf_opt_fini(void) {}


void worker_tf_opt_init(void)
{
    worker_cudnn_opt_init();
}

CUresult __pool_cuEventCreate(CUevent *phEvent, size_t count)
{
    int i;
    CUevent *desc;
    CUresult res = CUDA_SUCCESS;

    for (i = 0; i < count; i++) {
        desc = &phEvent[i];
        res = cuEventCreate(desc, 0);
        if (res != CUDA_SUCCESS)
            return res;
    }

    return res;
}

CUresult __pool_cuEventDestroy(CUevent *hEvent, size_t count)
{
    int i;
    CUresult res;

    for (i = 0; i < count; i++) {
        res = cuEventDestroy(hEvent[i]);
        if (res != CUDA_SUCCESS)
            return res;
    }

    return res;
}

CUresult __cuEventQuery(CUevent hEvent)
{
    return cuEventSynchronize(hEvent);
}
