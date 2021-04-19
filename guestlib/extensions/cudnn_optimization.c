#include "common/endpoint_lib.h"
#include "common/extensions/cudnn_optimization.h"

#include <glib.h>
#include <stdint.h>

GQueue *convolution_descriptor_pool;
GQueue *idle_convolution_descriptor_pool;
GQueue *pooling_descriptor_pool;
GQueue *idle_pooling_descriptor_pool;
GQueue *tensor_descriptor_pool;
GQueue *idle_tensor_descriptor_pool;
GQueue *filter_descriptor_pool;
GQueue *idle_filter_descriptor_pool;


void worker_cudnn_opt_init(void) {}

void guestlib_cudnn_opt_init(void)
{
    /* Pool descriptors */
    convolution_descriptor_pool = g_queue_new();
    idle_convolution_descriptor_pool = g_queue_new();

    pooling_descriptor_pool = g_queue_new();
    idle_pooling_descriptor_pool = g_queue_new();

    tensor_descriptor_pool = g_queue_new();
    idle_tensor_descriptor_pool = g_queue_new();

    filter_descriptor_pool = g_queue_new();
    idle_filter_descriptor_pool = g_queue_new();
}

void guestlib_cudnn_opt_fini(void)
{
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

