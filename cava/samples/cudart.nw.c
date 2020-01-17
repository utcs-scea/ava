ava_name("CUDA Runtime");
ava_version("10.0.0");
ava_identifier(CUDART);
ava_number(9);
ava_cflags(-I/usr/local/cuda-10.0/include -I../headers);
ava_libs(-L/usr/local/cuda-10.0/lib64 -lcudart -lcuda -lcublas -lcudnn);
ava_export_qualifier();

/**
 * Compile by
 * ./nwcc samples/cudart.nw.c -I /usr/local/cuda-10.0/include -I headers `pkg-config --cflags glib-2.0`
 */

ava_non_transferable_types {
    ava_handle;
}

size_t __args_index_0;
size_t __kernelParams_index_0;

ava_begin_utility;
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fatbinary.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <glib.h>
#include "cudart_nw_internal.h"

#include <stdio.h>

struct fatbin_wrapper {
    uint32_t magic;
    uint32_t seq;
    uint64_t ptr;
    uint64_t data_ptr;
};

struct kernel_arg {
    char is_handle;
    uint32_t size;
};

#define MAX_KERNEL_ARG 25
#define MAX_KERNEL_NAME_LEN 1024
#define MAX_ASYNC_BUFFER_NUM 16

struct fatbin_function {
    int argc;
    struct kernel_arg args[MAX_KERNEL_ARG];

    CUfunction cufunc;
    void *hostfunc;
    CUmodule module;
};
ava_end_utility;

ava_type(cudaError_t) {
    ava_success(CUDA_SUCCESS);
}

ava_type(CUresult) {
    ava_success(CUDA_SUCCESS);
}

/* Async buffer address list */
struct async_buffer_list {
    int num_buffers;
    void *buffers[MAX_ASYNC_BUFFER_NUM]; /* array of buffer addresses */
    size_t buffer_sizes[MAX_ASYNC_BUFFER_NUM];
};

typedef struct {
    /* argument types */
    GHashTable *fatbin_funcs;     /* for NULL, the hash table */
    int num_funcs;
    struct fatbin_function *func; /* for functions */

    /* global states */
    CUmodule cur_module;

    /* memory flags */
    int is_pinned;

    /* async buffers */
    struct async_buffer_list async_buffers;
} Metadata;

ava_register_metadata(Metadata);

ava_type(struct fatbin_wrapper) {
    struct fatbin_wrapper *ava_self;

    ava_field(magic);
    ava_field(seq);
    ava_field(ptr) {
        ava_type_cast(void *);
        ava_in; ava_buffer(((struct fatBinaryHeader *)ava_self->ptr)->headerSize + ((struct fatBinaryHeader *)ava_self->ptr)->fatSize);
        ava_lifetime_static;
    }
    ava_field(data_ptr) {
        ava_self->data_ptr = 0;
    }
}

ava_type(struct cudaDeviceProp);

ava_type(struct cudaPointerAttributes) {
    ava_field(devicePointer) ava_handle;
    ava_field(hostPointer) ava_opaque;
};

/* APIs needed for a minimal program */

char CUDARTAPI
__cudaInitModule(void **fatCubinHandle)
{
    ava_argument(fatCubinHandle) {
        ava_in; ava_buffer(1);
        ava_element ava_handle;
    }
}

ava_utility int __helper_cubin_num(void **cubin_handle) {
    int num = 0;
    while (cubin_handle[num] != NULL)
        num++;
    return num;
}

ava_utility void __helper_dump_fatbin(void *fatCubin,
                                    GHashTable **fatbin_funcs,
                                    int *num_funcs) {
    struct fatbin_wrapper *wp = fatCubin;
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)wp->ptr;

    /* Dump fat binary to a file */
    int fd;
    fd = open("/tmp/fatbin.cubin", O_WRONLY | O_TRUNC | O_CREAT, 0666);
    write(fd, (const void *)wp->ptr, fbh->headerSize + fbh->fatSize);
    close(fd);

    /* Execute cuobjdump and construct function information table */
    FILE *fp_pipe;
    char line[2048];
    int i, ordinal;
    size_t size;
    char name[MAX_KERNEL_NAME_LEN]; /* mangled name */
    struct fatbin_function *func;

    /* Create the hash table */
    if (*fatbin_funcs == NULL) {
        *fatbin_funcs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
        *num_funcs = 0;
    }

    /*  Open the command pipe for reading */
    fp_pipe = popen("/usr/local/cuda-10.0/bin/cuobjdump -elf /tmp/fatbin.cubin", "r");
    assert(fp_pipe);

    while (fgets(line, sizeof(line), fp_pipe) != NULL) {
        /* Search functions */
        if (strncmp(line, ".nv.info.", 9) == 0) {
            sprintf(name, line + 9, strlen(line) - 10);
            assert(strlen(line) - 10 < MAX_KERNEL_NAME_LEN);
            name[strlen(line) - 10] = '\0';
            DEBUG_PRINT("[%d] %s@\n", *num_funcs, name);

            if (g_hash_table_lookup(*fatbin_funcs, name) != NULL)
                continue;

            /* Create a new hash table entry */
            func = (struct fatbin_function *)g_malloc(sizeof(struct fatbin_function));
            memset(func, 0, sizeof(struct fatbin_function));

            // TODO: parse function name to determine whether the
            // arguments are handles

            /* Search parameters */
            func->argc = 0;
            while (fgets(line, sizeof(line), fp_pipe) != NULL) {
                i = 0;
                while (i < strlen(line) && isspace(line[i])) i++;
                /* Empty line means reaching the end of the function info */
                if (i == strlen(line)) break;

                if (strncmp(&line[i], "Attribute:", 10) == 0) {
                    i += 10;
                    while (i < strlen(line) && isspace(line[i])) i++;
                    if (strncmp(&line[i], "EIATTR_KPARAM_INFO", 18) == 0) {
                        /* Skip the format line */
                        fgets(line, sizeof(line), fp_pipe);
                        fgets(line, sizeof(line), fp_pipe);

                        /* Get ordinal and size */
                        i = 0;
                        while (i < strlen(line) && line[i] != 'O') i++;
                        sscanf(&line[i], "Ordinal\t: 0x%x", &ordinal);
                        while (i < strlen(line) && line[i] != 'S') i++;
                        sscanf(&line[i], "Size\t: 0x%lx", &size);

                        i = func->argc;
                        //DEBUG_PRINT("ordinal=%d, size=%lx\n", ordinal, size);
                        assert(ordinal < MAX_KERNEL_ARG);
                        func->args[ordinal].size = size;
                        ++(func->argc);
                    }
                }
            }

            ++(*num_funcs);

            /* Insert the function into hash table */
            g_hash_table_insert((*fatbin_funcs), g_strdup(name), (gpointer)func);
            //func = (struct fatbin_function *)g_hash_table_lookup(*fatbin_funcs, name);
        }
    }

    pclose(fp_pipe);
}

ava_utility void __helper_print_fatcubin_info(void *fatCubin, void **ret) {
    struct fatbin_wrapper *wp = fatCubin;
    printf("fatCubin_wrapper=%p, []={.magic=0x%X, .seq=%d, ptr=0x%lx, data_ptr=0x%lx}\n",
            fatCubin,
            wp->magic, wp->seq, wp->ptr, wp->data_ptr);
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)wp->ptr;
    printf("fatBinaryHeader={.magic=0x%X, version=%d, headerSize=0x%x, fatSize=0x%llx}\n",
            fbh->magic, fbh->version, fbh->headerSize, fbh->fatSize);
    char *fatBinaryEnd = (char *)(wp->ptr + fbh->headerSize + fbh->fatSize);
    printf("fatBin=0x%lx--0x%lx\n", wp->ptr, (int64_t)fatBinaryEnd);

    fatBinaryEnd = (char *)(wp->ptr);
    int i, j;
    for (i = 0; i < 100; i++)
        if (fatBinaryEnd[i] == 0x7F && fatBinaryEnd[i+1] == 'E' && fatBinaryEnd[i+2] == 'L') {
            printf("ELF header appears at 0x%d (%p): \n", i, (void *)wp->ptr + i);
            break;
        }
    for (j = i; j < i + 32; j++)
        printf("%.2X ", fatBinaryEnd[j] & 0xFF);
    printf("\n");

    printf("ret=%p\n", ret);
    printf("fatCubin=%p, *ret=%p\n", (void *)fatCubin, *ret);
}

ava_utility void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle) {
    int ret;
    if (ava_metadata(NULL)->cur_module == 0) {
        ret = cuInit(0);
        DEBUG_PRINT("ret=%d\n", ret);
        assert(ret == CUDA_SUCCESS && "CUDA driver init failed");
    }
    __cudaInitModule(handle);
    ret = cuModuleLoadData(&ava_metadata(NULL)->cur_module, (void *)fatCubin->ptr);
    DEBUG_PRINT("ret=%d, module=%lx\n", ret, (uintptr_t)ava_metadata(NULL)->cur_module);
    assert(ret == CUDA_SUCCESS && "Module load failed");
}

void** CUDARTAPI
__cudaRegisterFatBinary(void *fatCubin)
{
    ava_argument(fatCubin) {
        ava_type_cast(struct fatbin_wrapper *);
        ava_in; ava_buffer(1);
        ava_lifetime_static;
    }

    void **ret = (void **)ava_execute();
    ava_return_value {
        ava_out; ava_buffer(__helper_cubin_num(ret) + 1);
        ava_element {
            if (ret[ava_index] != NULL) ava_handle;
        }
        ava_allocates;
        ava_lifetime_manual;
    }

    __helper_dump_fatbin(fatCubin, &ava_metadata(NULL)->fatbin_funcs,
                        &ava_metadata(NULL)->num_funcs);

    if (ava_is_worker) {
        //__helper_print_fatcubin_info(fatCubin, ret);
        __helper_init_module(fatCubin, ret);
    }
}

ava_utility void __helper_unregister_fatbin(void **fatCubinHandle) {
    // free(fatCubinHandle);
    return;
}

void CUDARTAPI
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
    ava_disable_native_call;

    ava_argument(fatCubinHandle) {
        ava_in;
        /*
        ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
        ava_element {
            if (fatCubinHandle[ava_index] != NULL) ava_handle;
        }
        */
        ava_buffer(1);
        ava_element ava_handle;
        ava_deallocates;
    }

    if (ava_is_worker) {
        __helper_unregister_fatbin(fatCubinHandle);
    }
}

ava_utility void __helper_assosiate_function(GHashTable *funcs,
                                            struct fatbin_function **func,
                                            void *local,
                                            const char *deviceName) {
    if (*func != NULL) {
        DEBUG_PRINT("Function (%s) metadata (%p) already exists\n",
                deviceName, local);
        return;
    }

    *func = (struct fatbin_function *)g_hash_table_lookup(funcs, deviceName);
    assert(*func && "device function not found!");
}

ava_utility void __helper_register_function(struct fatbin_function *func,
                                            const char *hostFun,
                                            CUmodule module,
                                            const char *deviceName) {
    assert(func != NULL);
    /* Only register the first host function */
    if (func->hostfunc != NULL) return;

    CUresult ret = cuModuleGetFunction(&func->cufunc, module, deviceName);
    assert(ret == CUDA_SUCCESS);
    DEBUG_PRINT("register host func %lx -> device func %lx\n", (uintptr_t)hostFun, (uintptr_t)func->cufunc);
    func->hostfunc = (void *)hostFun;
    func->module = module;
}

ava_utility void __helper_parse_function_args(const char *name, struct kernel_arg *args)
{
    int i = 0, skip = 0;

    int argc = 0;
    if (strncmp(name, "_Z", 2)) abort();
    printf("kernel=%s\n", name);

    i = 2;
    while (i < strlen(name) && isdigit(name[i])) {
        skip = skip * 10 + name[i] - '0';
        i++;
    }

    i += skip;
    while (i < strlen(name)) {
        switch(name[i]) {
            case 'P':
                args[argc++].is_handle = 1;

                /* skip qualifiers */
                if (strchr("rVK", name[i+1]) != NULL)
                    i++;

                if (i + 1 < strlen(name) && (strchr("fijl", name[i+1]) != NULL))
                    i++;
                else if (i + 1 < strlen(name) && isdigit(name[i+1])) {
                    skip = 0;
                    while (i + 1 < strlen(name) && isdigit(name[i+1])) {
                        skip = skip * 10 + name[i+1] - '0';
                        i++;
                    }
                    i += skip;
                }
                else
                    abort();
                break;

            case 'f': /* float */
            case 'i': /* int */
            case 'j': /* unsigned int */
            case 'l': /* long */
                args[argc++].is_handle = 0;
                break;

            case 'S':
                args[argc++].is_handle = 1;
                while (i < strlen(name) && name[i] != '_') i++;
                break;

            case 'v':
                i = strlen(name);
                break;

            case 'r': /* restrict (C99) */
            case 'V': /* volatile */
            case 'K': /* const */
                break;

            default:
                abort();
        }
        i++;
    }

    for (i = 0; i < argc; i++) {
        DEBUG_PRINT("function arg#%d it is %sa handle\n", i, args[i].is_handle?"":"not ");
    }
}

void CUDARTAPI
__cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize)
{
    ava_disable_native_call;

    DEBUG_PRINT("register hostFun=%p, deviceFun=%s, deviceName=%s, thread_limit=%d, tid={%d,%d,%d}, bid={%d,%d,%d}, bDim={%d,%d,%d}, gDim={%d,%d,%d}\n",
            (void *)hostFun, deviceFun, deviceName, thread_limit,
            tid?tid->x:0, tid?tid->y:0, tid?tid->z:0,
            bid?bid->x:0, bid?bid->y:0, bid?bid->z:0,
            bDim?bDim->x:0, bDim?bDim->y:0, bDim?bDim->z:0,
            gDim?gDim->x:0, gDim?gDim->y:0, gDim?gDim->z:0);

    ava_argument(fatCubinHandle) {
        ava_in; ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
        ava_element {
            if (fatCubinHandle[ava_index] != NULL) ava_handle;
        }
    }

    ava_argument(hostFun) {
        ava_opaque;
    }

    ava_argument(deviceFun) {
        ava_in; ava_buffer(strlen(deviceFun) + 1);
    }

    ava_argument(deviceName) {
        ava_in; ava_buffer(strlen(deviceName) + 1);
    }

    __helper_assosiate_function(ava_metadata(NULL)->fatbin_funcs,
                &ava_metadata(hostFun)->func, (void *)hostFun,
                deviceName);

    ava_argument(tid) {
        ava_in; ava_buffer(1);
    }
    ava_argument(bid) {
        ava_in; ava_buffer(1);
    }
    ava_argument(bDim) {
        ava_in; ava_buffer(1);
    }
    ava_argument(gDim) {
        ava_in; ava_buffer(1);
    }
    ava_argument(wSize) {
        ava_in; ava_buffer(1);
    }

    if (ava_is_worker) {
        __helper_register_function(ava_metadata(hostFun)->func, hostFun,
                ava_metadata(NULL)->cur_module, deviceName);
    }
}

ava_begin_replacement;
void CUDARTAPI
__cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global)
{
}

void CUDARTAPI
__cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
#warning This API is called for CUDA 10.1 and 10.2, but it seems to be able to be ignored.
}
ava_end_replacement;

__host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3   gridDim,
                            dim3   blockDim,
                            size_t sharedMem, // CHECKME: default argument in header
                            void   *stream)
{
    ava_argument(stream) {
        ava_handle;
    }
}

cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3   *gridDim,
                           dim3   *blockDim,
                           size_t *sharedMem,
                           void   *stream)
{
    ava_argument(gridDim) {
        ava_out; ava_buffer(1);
    }
    ava_argument(blockDim) {
        ava_out; ava_buffer(1);
    }
    ava_argument(sharedMem) {
        ava_out; ava_buffer(1);
    }
    ava_argument(stream) {
        ava_type_cast(CUstream *);
        ava_out; ava_buffer(1);
        ava_element { ava_handle; }
    }
}

ava_utility void __helper_print_kernel_info(struct fatbin_function *func, void **args) {
    DEBUG_PRINT("function metadata (%p) for local %p, cufunc %p, argc %d\n",
            (void *)func, func->hostfunc, (void *)func->cufunc, func->argc);
    int i;
    for (i = 0; i < func->argc; i++) {
        DEBUG_PRINT("arg[%d] is %sa handle, size = %u, ptr = %p, content = %p\n", i,
                func->args[i].is_handle?"":"not ",
                func->args[i].size, args[i], *((void **)args[i]));
    }
}

ava_utility cudaError_t __helper_launch_kernel(struct fatbin_function *func,
                                            const void *hostFun,
                                            dim3 gridDim,
                                            dim3 blockDim,
                                            void **args,
                                            size_t sharedMem,
                                            cudaStream_t stream) {
    cudaError_t ret = (cudaError_t)CUDA_ERROR_PROFILER_ALREADY_STOPPED;

    if (func == NULL) return (cudaError_t)CUDA_ERROR_INVALID_PTX;

    if (func->hostfunc != hostFun) {
        fprintf(stderr, "search host func %p -> stored %p (device func %p)\n",
                hostFun, (void *)func->hostfunc, (void *)func->cufunc);
    }
    else {
        DEBUG_PRINT("matched host func %p -> device func %p\n", hostFun, (void *)func->cufunc);
    }
    __helper_print_kernel_info(func, args);
    ret = (cudaError_t)cuLaunchKernel(func->cufunc, gridDim.x, gridDim.y, gridDim.z,
                         blockDim.x, blockDim.y, blockDim.z,
                         sharedMem, (CUstream)stream,
                         args, NULL);

    return ret;
}

__host__ cudaError_t CUDARTAPI
cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
        size_t sharedMem, cudaStream_t stream)
{
    ava_disable_native_call;

    ava_argument(func) {
        ava_opaque;
    }

    ava_argument(args) {
        ava_in; ava_buffer(ava_metadata(func)->func->argc);
        ava_element {
            // FIXME: use the generated index name in the spec to
            // reference the outer loop's loop index at this moment.
            if (ava_metadata(func)->func->args[__args_index_0].is_handle) {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(func)->func->args[__args_index_0].size);
                //ava_element ava_handle;
            }
            else {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(func)->func->args[__args_index_0].size);
            }
        }
    }

    ava_argument(stream) {
        ava_handle;
    }

    cudaError_t ret;
    if (ava_is_worker) {
        ret = __helper_launch_kernel(ava_metadata(func)->func, func,
                                    gridDim, blockDim, args, sharedMem, stream);
#warning This will bypass the resource reporting routine.
        return ret;
    }
}

ava_begin_replacement;
__host__ cudaError_t CUDARTAPI
cudaMallocHost(void **ptr, size_t size)
{
    *ptr = malloc(size);
}

__host__ cudaError_t CUDARTAPI
cudaFreeHost(void *ptr)
{
    free(ptr);
}
ava_end_replacement;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMalloc(void **devPtr, size_t size)
{
    ava_argument(devPtr) {
        ava_out; ava_buffer(1);
        ava_element ava_opaque;
    }
}

__host__ cudaError_t CUDARTAPI
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    ava_argument(dst) {
        if (kind == cudaMemcpyHostToDevice) {
            ava_opaque;
        }
        else if (kind == cudaMemcpyDeviceToHost) {
            ava_out; ava_buffer(count);
        }
    }

    ava_argument(src) {
        if (kind == cudaMemcpyHostToDevice) {
            ava_in; ava_buffer(count);
        }
        else if (kind == cudaMemcpyDeviceToHost) {
            ava_opaque;
        }
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaFree(void *devPtr)
{
    ava_argument(devPtr) ava_opaque;
}

/* Rich set of APIs */

cudaError_t CUDARTAPI
cudaLaunch(const void *func)
{
    ava_unsupported;
}

cudaError_t CUDARTAPI
cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDevice(int *device)
{
    ava_argument(device) {
        ava_out; ava_buffer(1);
    }
}

__cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDeviceCount(int *count)
{
    ava_argument(count) {
        ava_out; ava_buffer(1);
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    ava_argument(prop) {
        ava_out; ava_buffer(1);
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    ava_argument(value) {
        ava_out; ava_buffer(1);
    }
}

__host__ cudaError_t CUDARTAPI
cudaDeviceReset(void);

__host__ cudaError_t CUDARTAPI
cudaSetDevice(int device);

__host__ cudaError_t CUDARTAPI
cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    ava_argument(symbol) {
        ava_opaque;
    }
    ava_argument(src) {
        ava_in; ava_buffer(count);
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    ava_argument(dst) {
        if (kind == cudaMemcpyHostToDevice) {
            ava_handle;
        }
        else if (kind == cudaMemcpyDeviceToHost) {
            ava_out; ava_buffer(count);
        }
    }

    ava_argument(src) {
        if (kind == cudaMemcpyHostToDevice) {
            ava_in; ava_buffer(count);
        }
        else if (kind == cudaMemcpyDeviceToHost) {
            ava_handle;
        }
    }

    ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaMemset(void *devPtr, int value, size_t count)
{
    ava_argument(devPtr) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr)
{
    ava_argument(attributes) {
        ava_out; ava_buffer(1);
    }
    ava_argument(ptr) {
        //ava_type_cast(CUdeviceptr);
        //ava_handle;
        ava_opaque;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceSynchronize(void);

__host__ cudaError_t CUDARTAPI
cudaEventCreate(cudaEvent_t *event)
{
    ava_argument(event) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    ava_argument(event) ava_handle;
    ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaEventQuery(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    ava_argument(ms) {
        ava_out; ava_buffer(1);
    }
    ava_argument(start) ava_handle;
    ava_argument(end) ava_handle;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventDestroy(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaEventSynchronize(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags)
{
    ava_unsupported;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetLastError(void);

__host__ __cudart_builtin__ const char* CUDARTAPI
cudaGetErrorString(cudaError_t error)
{
    const char *ret = ava_execute();
    ava_return_value {
        ava_out; ava_buffer(strlen(ret) + 1);
        ava_lifetime_static;
    }
}

__host__ cudaError_t CUDARTAPI
cudaMemGetInfo(size_t *_free, size_t *total)
{
    ava_argument(_free) {
        ava_out; ava_buffer(1);
    }
    ava_argument(total) {
        ava_out; ava_buffer(1);
    }
}

/* CUDA driver API */

CUresult CUDAAPI
cuInit(unsigned int Flags);

CUresult CUDAAPI
cuModuleGetFunction(CUfunction *hfunc,
                    CUmodule hmod,
                    const char *name)
{
    ava_argument(hfunc) {
        ava_out; ava_buffer(1);
    }
    ava_argument(name) {
        ava_in; ava_buffer(strlen(name) + 1);
    }

    ava_execute();
    __helper_parse_function_args(name, ava_metadata(*hfunc)->func->args);
}

ava_utility size_t __helper_fatbin_size(const void *cubin) {
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)cubin;
    return fbh->fatSize + fbh->headerSize;
}

CUresult CUDAAPI
cuModuleLoadData(CUmodule *module, const void *image)
{
    ava_argument(module) {
        ava_out; ava_buffer(1);
    }
    ava_argument(image) {
        ava_in; ava_buffer(__helper_fatbin_size(image));
    }
}

CUresult CUDAAPI
cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    ava_unsupported;
}

ava_utility size_t __helper_launch_extra_size(void **extra) {
    size_t size = 1;
    while (extra[size - 1] != CU_LAUNCH_PARAM_END)
        size++;
    return size;
}

CUresult CUDAAPI
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX,
               unsigned int gridDimY,
               unsigned int gridDimZ,
               unsigned int blockDimX,
               unsigned int blockDimY,
               unsigned int blockDimZ,
               unsigned int sharedMemBytes,
               CUstream hStream,
               void **kernelParams,
               void **extra)
{
    ava_argument(hStream) ava_handle;

    ava_argument(kernelParams) {
        ava_in; ava_buffer(ava_metadata(f)->func->argc);
        ava_element {
            // FIXME: use the generated index name in the spec to
            // reference the outer loop's loop index at this moment.
            if (ava_metadata(f)->func->args[__kernelParams_index_0].is_handle) {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(f)->func->args[__kernelParams_index_0].size);
                ava_element ava_handle;
            }
            else {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(f)->func->args[__kernelParams_index_0].size);
            }
        }
    }

    ava_argument(extra) {
        ava_in; ava_buffer(__helper_launch_extra_size(extra));
#warning The buffer size below states that every kernelParams[i] is 1 byte long.
        ava_element ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDeviceGetCount(int *count)
{
    ava_argument(count) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDeviceGet(CUdevice *device,
            int ordinal)
{
    ava_argument(device) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuCtxGetDevice(CUdevice *device)
{
    ava_argument(device) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDeviceGetName(char *name, int len, CUdevice dev)
{
    ava_argument(name) {
        ava_out; ava_buffer(len);
    }
}

CUresult CUDAAPI
cuDeviceGetUuid(CUuuid *uuid, CUdevice dev)
{
    ava_argument(uuid) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    ava_argument(pi) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
{
    ava_argument(flags) {
        ava_out; ava_buffer(1);
    }
    ava_argument(active) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);

CUresult CUDAAPI
cuCtxCreate(CUcontext *pctx,
            unsigned int flags,
            CUdevice dev)
{
    ava_argument(pctx) {
        ava_out; ava_element(ava_allocates); ava_buffer(1);
    }
}

CUresult CUDAAPI
cuCtxDestroy(CUcontext ctx)
{
    ava_argument(ctx) ava_deallocates;
}

CUresult CUDAAPI
cuCtxGetCurrent(CUcontext *pctx)
{
    ava_argument(pctx) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

CUresult CUDAAPI
cuCtxSetCurrent(CUcontext ctx)
{
    ava_argument(ctx) ava_handle;
}

CUresult CUDAAPI
cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    ava_argument(pctx) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

CUresult CUDAAPI
cuDevicePrimaryCtxRelease(CUdevice dev)
{
    ava_argument(dev) ava_handle;
}

CUresult CUDAAPI
cuCtxSynchronize(void);

CUresult
cuCtxPushCurrent(CUcontext ctx)
{
    ava_argument(ctx) ava_handle;
}

CUresult
cuCtxPopCurrent(CUcontext *pctx)
{
    ava_argument(pctx) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

CUresult CUDAAPI
cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    ava_unsupported;
}

CUresult CUDAAPI
cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    ava_unsupported;
}

CUresult CUDAAPI
cuCtxGetSharedMemConfig(CUsharedconfig *pConfig)
{
    ava_unsupported;
}

CUresult CUDAAPI
cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
    ava_argument(phStream) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

CUresult CUDAAPI
cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
{
    ava_argument(hStream) ava_handle;

    ava_argument(pctx) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

CUresult CUDAAPI
cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags)
{
    ava_unsupported;
}

CUresult CUDAAPI
cuStreamQuery(CUstream hStream)
{
    ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI
cuStreamDestroy(CUstream hStream)
{
    ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI
cuMemAlloc(CUdeviceptr *dptr,
           size_t bytesize)
{
    ava_argument(dptr) {
        ava_out; ava_buffer(1);
        ava_element { ava_opaque; ava_allocates; }
    }
}

ava_utility void *__helper_cu_mem_host_alloc_portable(size_t size)
{
    void *p = aligned_alloc(64, size);
    assert(p);
    return p;
}

ava_utility void __helper_cu_mem_host_free(void *ptr)
{
    free(ptr);
}

CUresult CUDAAPI
cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
{
    ava_argument(pp) {
        ava_out; ava_buffer(1);
        ava_element {
            ava_buffer(bytesize);
            ava_buffer_allocator(__helper_cu_mem_host_alloc_portable,
                    __helper_cu_mem_host_free);
            ava_lifetime_manual;
            ava_allocates;
            ava_no_copy;
        }
    }

    ava_execute();
    ava_metadata(*pp)->is_pinned = 1;
}

CUresult CUDAAPI
cuMemcpyHtoD(CUdeviceptr dstDevice,
             const void *srcHost,
             size_t ByteCount)
{
    ava_argument(dstDevice) ava_opaque;

    ava_argument(srcHost) {
        ava_in; ava_buffer(ByteCount);
        if (ava_metadata(srcHost)->is_pinned)
            ava_lifetime_manual;
    }
}

CUresult CUDAAPI
cuMemcpyDtoH(void *dstHost,
             CUdeviceptr srcDevice,
             size_t ByteCount)
{
    ava_argument(dstHost) {
        ava_out; ava_buffer(ByteCount);
        if (ava_metadata(dstHost)->is_pinned)
            ava_lifetime_manual;
    }

    ava_argument(srcDevice) ava_opaque;
}

CUresult CUDAAPI
cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                size_t ByteCount, CUstream hStream)
{
    ava_argument(dstDevice) ava_opaque;

    ava_argument(srcHost) {
        ava_in; ava_buffer(ByteCount);
        if (ava_metadata(srcHost)->is_pinned) {
            ava_lifetime_manual;
        }
        else {
            ava_lifetime_manual;
        }
#warning [issue#65] deallocate the buffer for async memory copy at the \
        synchronization point (ava_lifetime_sync).
    }

    ava_argument(hStream) ava_handle;
}

/**
 * Saves the async buffer information into the list inside the stream's
 * metadata.
 */
ava_utility void __helper_register_async_buffer(struct async_buffer_list *buffers,
                                                void *buffer, size_t size) {
    assert(buffers->num_buffers < MAX_ASYNC_BUFFER_NUM);
    int idx = (buffers->num_buffers)++;
    DEBUG_PRINT("Register async buffer [%d] address = %p, size = %ld\n", idx, buffer, size);
    buffers->buffers[idx] = buffer;
    buffers->buffer_sizes[idx] = size;
}

CUresult CUDAAPI
cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
        size_t ByteCount, CUstream hStream)
{
    __helper_register_async_buffer(&ava_metadata(hStream)->async_buffers,
                                dstHost, ByteCount);

    ava_argument(dstHost) {
        ava_out; ava_buffer(ByteCount);
        if (ava_metadata(dstHost)->is_pinned) {
            ava_lifetime_manual;
        }
        else {
            ava_lifetime_manual;
        }
#warning [issue#65] deallocate the buffer for async memory copy at the \
        synchronization point (ava_lifetime_sync).
    }

    ava_argument(srcDevice) ava_opaque;
    ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI
cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    ava_argument(dstDevice) ava_opaque;
}

CUresult CUDAAPI
cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    ava_argument(dstDevice) ava_opaque;
}

CUresult CUDAAPI
cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    ava_async;
    ava_argument(dstDevice) ava_opaque;
    ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI
cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    ava_async;
    ava_argument(dstDevice) ava_opaque;
    ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI
cuMemFreeHost(void *p)
{
    ava_metadata(p)->is_pinned = 0;
    ava_deallocates;
}

CUresult CUDAAPI
cuDriverGetVersion(int *driverVersion)
{
    ava_argument(driverVersion) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    ava_unsupported;
}

CUresult CUDAAPI
cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    ava_argument(bytes) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuMemGetInfo(size_t *_free, size_t *total)
{
    ava_argument(_free) {
        ava_out; ava_buffer(1);
    }
    ava_argument(total) {
        ava_out; ava_buffer(1);
    }
}

CUresult CUDAAPI
cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev)
{
    ava_argument(pciBusId) {
        ava_out; ava_buffer(len);
    }
}

CUresult CUDAAPI
cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
    ava_argument(phEvent) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

CUresult CUDAAPI
cuEventQuery(CUevent hEvent)
{
    ava_argument(hEvent) ava_handle;
}

CUresult CUDAAPI
cuEventRecord(CUevent hEvent, CUstream hStream)
{
    ava_argument(hEvent) ava_handle;
    ava_argument(hStream) ava_handle;
}

CUresult CUDAAPI
cuEventSynchronize(CUevent hEvent) {
    ava_argument(hEvent) ava_handle;
}

CUresult CUDAAPI
cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
    ava_argument(pMilliseconds) {
        ava_out; ava_buffer(1);
    }
    ava_argument(hStart) ava_handle;
    ava_argument(hEnd) ava_handle;
}

CUresult
cuEventDestroy(CUevent hEvent)
{
    ava_argument(hEvent) ava_handle;
}

ava_utility struct async_buffer_list *__helper_load_async_buffer_list(
        struct async_buffer_list *buffers) {
    if (buffers->num_buffers == 0) return NULL;

    DEBUG_PRINT("Load %d async buffers\n", buffers->num_buffers);
    struct async_buffer_list *new_copy =
        (struct async_buffer_list *)malloc(sizeof(struct async_buffer_list));
    memcpy(new_copy, buffers, sizeof(struct async_buffer_list));
    memset(buffers, 0, sizeof(struct async_buffer_list));

    return new_copy;
}

CUresult CUDAAPI
cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    ava_argument(hStream) ava_handle;
    ava_argument(hEvent) ava_handle;

#warning Fix the update of the buffers that are copied asynchronously.
    struct async_buffer_list *async_buffers;
    async_buffers = __helper_load_async_buffer_list(
            &ava_metadata(hStream)->async_buffers);

    /*
    ava_implicit_argument
    int num_buffers = async_buffers == NULL ? 0 : async_buffers->num_buffers;

    ava_implicit_argument
    size_t *buffer_sizes = async_buffers == NULL ? NULL : async_buffers->buffer_sizes;
    ava_argument(buffer_sizes) {
        ava_in; ava_buffer(num_buffers);
    }

    ava_implicit_argument
    void **buffers = async_buffers == NULL ? NULL : async_buffers->buffers;
    ava_argument(buffers) {
        ava_in; ava_buffer(num_buffers);
        ava_element {
            ava_out;
            ava_buffer(buffer_sizes[ava_index]);
        }
    }
    */

    if (async_buffers != NULL)
        free(async_buffers);
}

CUresult
cuGetExportTable(const void **ppExportTable, const CUuuid * pExportTableId)
{
    ava_unsupported;
}

/* CUDABLAS API */
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCreate(cublasHandle_t *handle)
{
    ava_argument(handle) {
        ava_out; ava_buffer(1);
        ava_element { ava_handle; }
    }
}

cublasStatus_t CUBLASWINAPI
cublasSetMatrix (int rows, int cols, int elemSize,
                const void *A, int lda,
                void *B, int ldb)
{
    ava_argument(A) {
        ava_in; ava_buffer(rows * cols * elemSize);
    }

    ava_argument(B) {
        ava_handle;
    }
}

cublasStatus_t CUBLASWINAPI
cublasGetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda,
                void *B, int ldb)
{
    ava_argument(A) {
        ava_handle;
    }

    ava_argument(B) {
        ava_out; ava_buffer(rows * cols * elemSize);
    }
}

ava_begin_replacement;
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode)
{
    /* XXX seems ok for tensorflow but might be wrong !FIXME */
    *mode = 0;
    return CUBLAS_STATUS_SUCCESS;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode)
{
    /* XXX seems ok for tensorflow but might be wrong ! FIXME */
    assert(mode == 0);
    return CUBLAS_STATUS_SUCCESS;
}
ava_end_replacement;


CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSgemm_v2 (cublasHandle_t handle, cublasOperation_t transa,
					 cublasOperation_t transb, int m, int n, int k,
					 const float *alpha, /* host or device pointer */
					 const float *A, int lda, const float *B, int ldb,
					 const float *beta, /* host or device pointer */
					 float *C, int ldc)
{
    ava_async;
    ava_argument(handle) ava_handle;
    ava_argument(transa) ava_opaque;
	ava_argument(transb) ava_opaque;
    ava_argument(A) ava_handle;
    ava_argument(B) ava_handle;
    ava_argument(C) ava_handle;
	 /* XXX I _think_ these are always device pointers for tensorflow ! */
    ava_argument(alpha) { ava_in; ava_buffer(1); }
    ava_argument(beta)  { ava_in; ava_buffer(1); }
}


CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    ava_argument(handle) ava_handle;
    ava_argument(streamId) ava_handle;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDestroy(cublasHandle_t handle)
{
    ava_argument(handle) ava_handle;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSscal(cublasHandle_t handle,
            int n,
            const float *alpha,  /* host or device pointer */
            float *x,
            int incx)
{
    ava_argument(handle) ava_handle;
    ava_argument(alpha) {
        ava_in; ava_buffer(1);
    }
    ava_argument(x) ava_handle;
}

/***** CUDNN (OOF) ******/

cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
                                        cudnnBatchNormMode_t mode,
                                        const void *alpha, /* alpha[0] = result blend factor */
                                        const void *beta,  /* beta[0] = dest layer blend factor */
                                        const cudnnTensorDescriptor_t xDesc,
                                        const void *x, /* NxCxHxW */
                                        const cudnnTensorDescriptor_t yDesc,
                                        void *y, /* NxCxHxW */
                                        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        const void *bnScale,
                                        const void *bnBias,
                                        const void *estimatedMean,
                                        const void *estimatedVariance,
                                        double epsilon)
{
   ava_async;
   ava_argument(handle) ava_handle;
   ava_argument(alpha) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(beta) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(xDesc) ava_handle;
   ava_argument(x) ava_handle;
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_handle;
   ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
   ava_argument(bnScale) ava_handle;
   ava_argument(bnBias) ava_handle;
   ava_argument(estimatedMean) ava_handle;
   ava_argument(estimatedVariance) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnConvolutionForward(cudnnHandle_t handle,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo,
                        void *workSpace,
                        size_t workSpaceSizeInBytes,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y)
{
   ava_async;
   ava_argument(handle) ava_handle;
   ava_argument(alpha) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(beta) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(xDesc) ava_handle;
   ava_argument(x) ava_handle;
   ava_argument(wDesc) ava_handle;
   ava_argument(w) ava_handle;
   ava_argument(convDesc) ava_handle;
   ava_argument(workSpace) ava_handle;
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnCreate(cudnnHandle_t *handle)
{
   ava_argument(handle) {
      ava_out; ava_buffer(1);
      ava_element ava_handle;
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc)
{
   ava_argument(convDesc) {
      ava_out; ava_buffer(1);
      ava_element ava_handle;
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc)
{
   ava_argument(filterDesc) {
      ava_out; ava_buffer(1);
      ava_element ava_handle;
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc)
{
   ava_argument(poolingDesc) {
      ava_out; ava_buffer(1);
      ava_element ava_handle;
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc)
{
   ava_argument(tensorDesc) {
      ava_out; ava_buffer(1);
      ava_element ava_handle;
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc)
{
   ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc)
{
   ava_argument(filterDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc)
{
   ava_argument(poolingDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc)
{
   ava_argument(tensorDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle,
                                                         cudnnBatchNormMode_t mode,
                                                         cudnnBatchNormOps_t bnOps,
                                                         const cudnnTensorDescriptor_t xDesc,
                                                         const cudnnTensorDescriptor_t zDesc,
                                                         const cudnnTensorDescriptor_t yDesc,
                                                         const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                         const cudnnActivationDescriptor_t activationDesc,
                                                         size_t *sizeInBytes)
{
   ava_argument(handle) ava_handle;
   ava_argument(xDesc) ava_handle;
   ava_argument(zDesc) ava_handle;
   ava_argument(yDesc) ava_handle;
   ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
   ava_argument(activationDesc) ava_handle;
   ava_argument(sizeInBytes) {
      ava_out; ava_buffer(1);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                        const cudnnTensorDescriptor_t xDesc,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const cudnnConvolutionDescriptor_t convDesc,
                                        const cudnnTensorDescriptor_t yDesc,
                                        cudnnConvolutionFwdAlgo_t algo,
                                        size_t *sizeInBytes)
{
   ava_argument(handle) ava_handle;
   ava_argument(xDesc) ava_handle;
   ava_argument(wDesc) ava_handle;
   ava_argument(convDesc) ava_handle;
   ava_argument(yDesc) ava_handle;
   ava_argument(sizeInBytes) {
      ava_out; ava_buffer(1);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnGetProperty(libraryPropertyType type, int *value)
{
   ava_argument(value) {
      ava_out; ava_buffer(1);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnPoolingForward(cudnnHandle_t handle,
                    const cudnnPoolingDescriptor_t poolingDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
{
   ava_async;
   ava_argument(handle) ava_handle;
   ava_argument(poolingDesc) ava_handle;
   ava_argument(alpha) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(xDesc) ava_handle;
   ava_argument(x) ava_handle;
   ava_argument(beta) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount)
{
   ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType)
{
   ava_argument(convDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                int arrayLength, /* nbDims-2 size */
                                const int padA[],
                                const int filterStrideA[],
                                const int dilationA[],
                                cudnnConvolutionMode_t mode,
                                cudnnDataType_t computeType) /* convolution data type */
{
   ava_argument(convDesc) ava_handle;
   ava_argument(padA) {
      ava_in; ava_buffer(arrayLength);
   }
   ava_argument(filterStrideA) {
      ava_in; ava_buffer(arrayLength);
   }
   ava_argument(dilationA) {
      ava_in; ava_buffer(arrayLength);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format,
                           int nbDims,
                           const int filterDimA[])
{
   ava_argument(filterDesc) ava_handle;
   ava_argument(filterDimA) {
      ava_in; ava_buffer(nbDims);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            const cudnnPoolingMode_t mode,
                            const cudnnNanPropagation_t maxpoolingNanOpt,
                            int nbDims,
                            const int windowDimA[],
                            const int paddingA[],
                            const int strideA[])
{
   ava_argument(poolingDesc) ava_handle;
   ava_argument(windowDimA) {
      ava_in; ava_buffer(nbDims);
   }
   ava_argument(paddingA) {
      ava_in; ava_buffer(nbDims);
   }
   ava_argument(strideA) {
      ava_in; ava_buffer(nbDims);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId)
{
    ava_argument(handle) ava_handle;
    ava_argument(streamId) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t dataType,
                           int nbDims,
                           const int dimA[],
                           const int strideA[])
{
    ava_argument(tensorDesc) ava_handle;
   ava_argument(dimA) {
      ava_in; ava_buffer(nbDims);
   }
   ava_argument(strideA) {
      ava_in; ava_buffer(nbDims);
   }
}

cudnnStatus_t CUDNNWINAPI
cudnnPoolingBackward(cudnnHandle_t handle,
                     const cudnnPoolingDescriptor_t poolingDesc,
                     const void *alpha,
                     const cudnnTensorDescriptor_t yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t dyDesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t xDesc,
                     const void *x,
                     const void *beta,
                     const cudnnTensorDescriptor_t dxDesc,
                     void *dx)
{
   ava_argument(handle) ava_handle;
   ava_argument(poolingDesc) ava_handle;
   ava_argument(alpha) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_handle;
   ava_argument(dyDesc) ava_handle;
   ava_argument(dy) ava_handle;
   ava_argument(xDesc) ava_handle;
   ava_argument(x) ava_handle;
   ava_argument(beta) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(dxDesc) ava_handle;
   ava_argument(dx) ava_handle;
}
