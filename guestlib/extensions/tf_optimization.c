#include "common/endpoint_lib.h"
#include "common/extensions/cmd_batching.h"
#include "common/extensions/tf_optimization.h"

#include <glib.h>
#include <stdint.h>

GQueue *call_configuration_stack;
GTree *gpu_address_set;

GQueue *convolution_descriptor_pool;
GQueue *idle_convolution_descriptor_pool;
GQueue *pooling_descriptor_pool;
GQueue *idle_pooling_descriptor_pool;
GQueue *tensor_descriptor_pool;
GQueue *idle_tensor_descriptor_pool;
GQueue *filter_descriptor_pool;
GQueue *idle_filter_descriptor_pool;
GQueue *cu_event_pool;
GQueue *idle_cu_event_pool;

cudaError_t cuda_last_error;

gint gpu_address_range_cmp(gconstpointer r1, gconstpointer r2, gpointer user_data)
{
    long diff = ((uintptr_t)r1 - (uintptr_t)r2);
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

void guestlib_tf_opt_init(void)
{
    /* Emulate the call configuration stack */
    call_configuration_stack = g_queue_new();

    /* Save allocated GPU memory addresses */
    gpu_address_set = g_tree_new_full(gpu_address_range_cmp, NULL, NULL, g_free);

    /* Pool descriptors */
    convolution_descriptor_pool = g_queue_new();
    idle_convolution_descriptor_pool = g_queue_new();

    pooling_descriptor_pool = g_queue_new();
    idle_pooling_descriptor_pool = g_queue_new();

    tensor_descriptor_pool = g_queue_new();
    idle_tensor_descriptor_pool = g_queue_new();

    filter_descriptor_pool = g_queue_new();
    idle_filter_descriptor_pool = g_queue_new();

    cu_event_pool = g_queue_new();
    idle_cu_event_pool = g_queue_new();

    /* API batch */
    nw_global_cmd_batch = cmd_batch_thread_init();
}

void guestlib_tf_opt_fini(void)
{
    g_queue_free(call_configuration_stack);
    g_tree_destroy(gpu_address_set);

    /* Free descriptors */
    free_convolution_descriptor_pool(convolution_descriptor_pool);
    free_convolution_descriptor_pool(idle_convolution_descriptor_pool);
    g_queue_free(convolution_descriptor_pool);
    g_queue_free(idle_convolution_descriptor_pool);

    free_pooling_descriptor_pool(pooling_descriptor_pool);
    free_pooling_descriptor_pool(idle_pooling_descriptor_pool);
    g_queue_free(pooling_descriptor_pool);
    g_queue_free(idle_pooling_descriptor_pool);

    free_tensor_descriptor_pool(tensor_descriptor_pool);
    free_tensor_descriptor_pool(idle_tensor_descriptor_pool);
    g_queue_free(tensor_descriptor_pool);
    g_queue_free(idle_tensor_descriptor_pool);

    free_filter_descriptor_pool(filter_descriptor_pool);
    free_filter_descriptor_pool(idle_filter_descriptor_pool);
    g_queue_free(filter_descriptor_pool);
    g_queue_free(idle_filter_descriptor_pool);

    free_cu_event_pool(cu_event_pool);
    free_cu_event_pool(idle_cu_event_pool);
    g_queue_free(cu_event_pool);
    g_queue_free(idle_cu_event_pool);

    cmd_batch_thread_fini(nw_global_cmd_batch);
}

int free_convolution_descriptor_pool(GQueue *pool)
{
    gpointer element;
    cudnnConvolutionDescriptor_t *desc;
    int i = 0;

    if (g_queue_is_empty(pool))
        return CUDNN_STATUS_SUCCESS;

    desc = (cudnnConvolutionDescriptor_t *)
        malloc(sizeof(cudnnConvolutionDescriptor_t) * pool->length);

    while ((element = g_queue_pop_head(pool))) {
        desc[i++] = (cudnnConvolutionDescriptor_t)element;
    }

    return __pool_cudnnDestroyConvolutionDescriptor(desc, i);
}

int free_pooling_descriptor_pool(GQueue *pool)
{
    gpointer element;
    cudnnPoolingDescriptor_t *desc;
    int i = 0;

    if (g_queue_is_empty(pool))
        return CUDNN_STATUS_SUCCESS;

    desc = (cudnnPoolingDescriptor_t *)
        malloc(sizeof(cudnnPoolingDescriptor_t) * pool->length);

    while ((element = g_queue_pop_head(pool))) {
        desc[i++] = (cudnnPoolingDescriptor_t)element;
    }

    return __pool_cudnnDestroyPoolingDescriptor(desc, i);
}

int free_tensor_descriptor_pool(GQueue *pool)
{
    gpointer element;
    cudnnTensorDescriptor_t *desc;
    int i = 0;

    if (g_queue_is_empty(pool))
        return CUDNN_STATUS_SUCCESS;

    desc = (cudnnTensorDescriptor_t *)
        malloc(sizeof(cudnnTensorDescriptor_t) * pool->length);

    while ((element = g_queue_pop_head(pool))) {
        desc[i++] = (cudnnTensorDescriptor_t)element;
    }

    return __pool_cudnnDestroyTensorDescriptor(desc, i);
}

int free_filter_descriptor_pool(GQueue *pool)
{
    gpointer element;
    cudnnFilterDescriptor_t *desc;
    int i = 0;

    if (g_queue_is_empty(pool))
        return CUDNN_STATUS_SUCCESS;

    desc = (cudnnFilterDescriptor_t *)
        malloc(sizeof(cudnnFilterDescriptor_t) * pool->length);

    while ((element = g_queue_pop_head(pool))) {
        desc[i++] = (cudnnFilterDescriptor_t)element;
    }

    return __pool_cudnnDestroyFilterDescriptor(desc, i);
}

int free_cu_event_pool(GQueue *pool)
{
    gpointer element;
    CUevent *desc;
    int i = 0;

    if (g_queue_is_empty(pool))
        return CUDA_SUCCESS;

    desc = (CUevent *)malloc(sizeof(CUevent) * pool->length);

    while ((element = g_queue_pop_head(pool))) {
        desc[i++] = (CUevent)element;
    }

    return __pool_cuEventDestroy(desc, i);
}
