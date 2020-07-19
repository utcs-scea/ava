ava_name("CUDA Runtime for ONNX");
ava_version("10.1.0");
ava_identifier(ONNX_DUMP);
ava_number(10);
ava_cflags(-I/usr/local/cuda-10.1/include -I../headers);
ava_libs(-L/usr/local/cuda-10.1/lib64 -lcudart -lcuda -lcublas -lcudnn -lcufft -lcurand -lcusparse -lcusolver);
ava_export_qualifier();

/**
 * The spec is used to dump the fat binaries and CUDA functions from
 * ONNXruntime library.
 * Compile by
 * ./nwcc samples/onnxruntime/onnx_dump.c -I /usr/local/cuda-10.1/include -I headers `pkg-config --cflags glib-2.0`
 *
 * Dependencies:
 * CUDA 10.1, cuDNN 7.6.5
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
#include <cublas_api.h>
#include <cudnn.h>
#include <curand.h>
#include <cufft.h>
#include <cusparse.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <glib.h>
#include "cudart_nw_internal.h"
#include "cublas_cpp.h"
#include <unistd.h>
#include <errno.h>

#include <stdio.h>

#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v) \
        = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */

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

#define MAX_KERNEL_ARG 30
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
    ava_success(cudaSuccess);
}

ava_type(cublasStatus_t) {
    ava_success(CUBLAS_STATUS_SUCCESS);
}

ava_type(cudnnStatus_t ) {
    ava_success(CUDNN_STATUS_SUCCESS);
}

ava_type(CUresult) {
    ava_success(CUDA_SUCCESS);
}

ava_type(curandStatus_t) {
    ava_success(CURAND_STATUS_SUCCESS);
}

ava_type(cufftResult) {
    ava_success(CUFFT_SUCCESS);
}

ava_type(cusparseStatus_t) {
    ava_success(CUSPARSE_STATUS_SUCCESS);
}

ava_type(cusolverStatus_t) {
    ava_success(CUSOLVER_STATUS_SUCCESS);
}

/* Async buffer address list */
struct async_buffer_list {
    int num_buffers;
    void *buffers[MAX_ASYNC_BUFFER_NUM]; /* array of buffer addresses */
    size_t buffer_sizes[MAX_ASYNC_BUFFER_NUM];
};

typedef struct {
    int num_fatbins;
    int fd_functions;

    /* argument types */
    GHashTable *fatbin_funcs;     /* for NULL, the hash table */
    int num_funcs;
    struct fatbin_function *func; /* for functions */

    /* global states */
    CUmodule cur_module;
    int cuinit_called;

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
    int fd, ret;

    /* Increase fatbin counter */
    static int fatbin_num = 0;
    fatbin_num++;
    if (ava_is_worker) {
        char* file_name = "/tmp/fatbin-info.ava";
        fd = open(file_name, O_RDWR | O_CREAT, 0666);
        if (fd == -1) {
            fprintf(stderr, "open %s [errno=%d, errstr=%s] at %s:%d",
                file_name, errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT("Fatbinary counter = %d\n", fatbin_num);
        ret = write(fd, (const void *)&fatbin_num, sizeof(int));
        if (ret == -1) {
            fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        ret = lseek(fd, 0, SEEK_END);
        if (ret == -1) {
            fprintf(stderr, "lseek [errno=%d, errstr=%s] at %s:%d",
                errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        ret = write(fd, (const void *)wp, sizeof(struct fatbin_wrapper));
        if (ret == -1) {
            fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        close(fd);
    }

    /* Dump fat binary to a file */
    char fatbin_filename[32];
    if (ava_is_worker) {
        sprintf(fatbin_filename, "/tmp/fatbin-%d.ava", ava_metadata(NULL)->num_fatbins);
        fd = open(fatbin_filename, O_WRONLY | O_TRUNC | O_CREAT, 0666);
        if (fd == -1) {
            fprintf(stderr, "open %s [errno=%d, errstr=%s] at %s:%d",
                fatbin_filename, errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT("Dump fatbinary to %s\n", fatbin_filename);
        ret = write(fd, (const void *)wp->ptr, fbh->headerSize + fbh->fatSize);
        if (ret == -1) {
            fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        close(fd);
    }

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

    /* Add separator to the functions of different fatbinaries */
    if (ava_is_worker) {
        if (ava_metadata(NULL)->fd_functions != 0) {
            size = 0;
            ret = write(ava_metadata(NULL)->fd_functions, (const void *)&size, sizeof(size_t));
            if (ret == -1) {
                fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                    errno, strerror(errno), __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
        }
    }

    /*  Open the command pipe for reading */
    char pip_command[80];
    sprintf(pip_command, "/usr/local/cuda-10.1/bin/cuobjdump -elf /tmp/fatbin-%d.ava",
            ava_metadata(NULL)->num_fatbins);
    fp_pipe = popen(pip_command, "r");
    assert(fp_pipe);

    /* Open function argument dump file */
    int function_arg_fd;
    char function_arg_filename[32];
    if (ava_is_worker) {
        sprintf(function_arg_filename, "/tmp/function_arg-%d.ava", ava_metadata(NULL)->num_fatbins);
        function_arg_fd = open(function_arg_filename, O_WRONLY | O_TRUNC | O_CREAT, 0666);
        if (function_arg_fd == -1) {
            fprintf(stderr, "open %s [errno=%d, errstr=%s] at %s:%d",
                function_arg_filename, errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT("Dump function argument info to %s\n", function_arg_filename);
    }

    while (fgets(line, sizeof(line), fp_pipe) != NULL) {
        /* Search functions */
        if (strncmp(line, ".nv.info._Z", 11) == 0) {
            sprintf(name, line + 9, strlen(line) - 10);
            assert(strlen(line) - 10 < MAX_KERNEL_NAME_LEN);
            name[strlen(line) - 10] = '\0';
            DEBUG_PRINT("[%d] %s@\n", *num_funcs, name);

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

            /* Dump the function argument sizes to file */
            if (ava_is_worker) {
                size = strlen(name) + 1;
                ret = write(function_arg_fd, (void *)&size, sizeof(size_t));
                if (ret == -1) {
                    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                        errno, strerror(errno), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }
                ret = write(function_arg_fd, (void *)name, size);
                if (ret == -1) {
                    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                        errno, strerror(errno), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }
                ret = write(function_arg_fd, (void *)func, sizeof(struct fatbin_function));
                if (ret == -1) {
                    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d",
                        errno, strerror(errno), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }
            }

            /* Insert the function into hash table */
            if (g_hash_table_lookup(*fatbin_funcs, name) != NULL)
                g_free(func);
            else
                g_hash_table_insert((*fatbin_funcs), g_strdup(name), (gpointer)func);
            //func = (struct fatbin_function *)g_hash_table_lookup(*fatbin_funcs, name);
        }
    }

    if (ava_is_worker)
        close(function_arg_fd);

    pclose(fp_pipe);
    ++(ava_metadata(NULL)->num_fatbins);
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
    if (ava_metadata(NULL)->cuinit_called == 0) {
        ret = cuInit(0);
        DEBUG_PRINT("ret=%d\n", ret);
        assert(ret == CUDA_SUCCESS && "CUDA driver init failed");
        ava_metadata(NULL)->cuinit_called = 1;
    }
    __cudaInitModule(handle);
    ava_metadata(NULL)->cur_module = NULL;
    ret = cuModuleLoadData(&ava_metadata(NULL)->cur_module, (void *)fatCubin->ptr);
    (void)ret;
    DEBUG_PRINT("ret=%d, module=%lx\n", ret, (uintptr_t)ava_metadata(NULL)->cur_module);
    assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
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
        ava_in; ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
        ava_element {
            if (fatCubinHandle[ava_index] != NULL) ava_handle;
        }
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
    if (*func == NULL) {
        fprintf(stderr, "device name is %s\n", deviceName);
    }
    assert(*func && "device function not found!");
}

ava_utility void __helper_register_function(struct fatbin_function *func,
                                            const char *hostFun,
                                            CUmodule module,
                                            const char *deviceName) {
    /* Empty fatbinary */
    if (!module)
        return;

    assert(func != NULL);
    /* Only register the first host function */
    if (func->hostfunc != NULL) return;

    CUresult ret = cuModuleGetFunction(&func->cufunc, module, deviceName);
    assert(ret == CUDA_SUCCESS);
    (void)ret;
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

ava_utility void __helper_dump_cuda_function(
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize) {
    int fd = ava_metadata(NULL)->fd_functions;
    if (fd == 0) {
        fd = open("/tmp/fatfunction.ava", O_WRONLY | O_TRUNC | O_CREAT, 0666);
        if (fd == -1) {
            fprintf(stderr, "open /tmp/fatfunction.ava [errno=%d, errstr=%s] at %s:%d",
                errno, strerror(errno), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        ava_metadata(NULL)->fd_functions = fd;
    }

    size_t size;
    int exists;
    size = strlen(deviceFun) + 1;
    write(fd, (const void *)&size, sizeof(size_t));
    write(fd, (const void *)deviceFun, size);
    size = strlen(deviceName) + 1;
    write(fd, (const void *)&size, sizeof(size_t));
    write(fd, (const void *)deviceName, size);
    write(fd, (const void *)&thread_limit, sizeof(int));
    exists = (tid != NULL);
    write(fd, (const void *)&exists, sizeof(int));
    if (exists)
        write(fd, (const void *)tid, sizeof(uint3));
    exists = (bid != NULL);
    write(fd, (const void *)&exists, sizeof(int));
    if (exists)
        write(fd, (const void *)bid, sizeof(uint3));
    exists = (bDim != NULL);
    write(fd, (const void *)&exists, sizeof(int));
    if (exists)
        write(fd, (const void *)bDim, sizeof(dim3));
    exists = (gDim != NULL);
    write(fd, (const void *)&exists, sizeof(int));
    if (exists)
        write(fd, (const void *)gDim, sizeof(dim3));
    exists = (wSize != NULL);
    write(fd, (const void *)&exists, sizeof(int));
    if (exists)
        write(fd, (const void *)wSize, sizeof(int));
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

    if (ava_is_worker)
        __helper_dump_cuda_function(deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

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
        return ret;
    }
}

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

    ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaMemset(void *devPtr, int value, size_t count)
{
    ava_argument(devPtr) ava_opaque;
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

/*
ava_callback_decl void __callback_cuda_stream_add_callback(
        cudaStream_t stream,  cudaError_t status, void*  userData) {
    ava_argument(stream) ava_handle;
    ava_argument(userData) {
        ava_userdata;
    }
}

__host__ cudaError_t CUDARTAPI
cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags)
{
    ava_argument(stream) ava_handle;
    ava_argument(callback) ava_callback(__callback_cuda_stream_add_callback);
}
*/

ava_begin_replacement;
__host__ cudaError_t CUDARTAPI
cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags)
{
    return cudaSuccess;
}
ava_end_replacement;

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
                ava_element ava_opaque;
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
    /*
    __helper_register_async_buffer(&ava_metadata(hStream)->async_buffers,
                                dstHost, ByteCount);
     */

    ava_argument(dstHost) {
#warning async buffers need to be no_copy
        // ava_no_copy;
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

#warning Force synchronization of async buffers
    ava_execute();
    if (ava_is_worker) {
        cudaStreamSynchronize(hStream);
    }
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
    int size = sizeof(struct async_buffer_list);
    struct async_buffer_list *new_copy =
        (struct async_buffer_list *)malloc(size);
    if (new_copy == NULL) {
        fprintf(stderr, "malloc size=%d [errno=%d, errstr=%s] at %s:%d",
            size, errno, strerror(errno), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    memcpy(new_copy, buffers, sizeof(struct async_buffer_list));
    memset(buffers, 0, sizeof(struct async_buffer_list));

    return new_copy;
}

CUresult CUDAAPI
cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    ava_argument(hStream) ava_handle;
    ava_argument(hEvent) ava_handle;

    /*
#warning Fix the update of the buffers that are copied asynchronously.
    struct async_buffer_list *async_buffers;
    async_buffers = __helper_load_async_buffer_list(
            &ava_metadata(hStream)->async_buffers);

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

    if (async_buffers != NULL)
        free(async_buffers);
    */
}

CUresult
cuGetExportTable(const void **ppExportTable, const CUuuid * pExportTableId)
{
    ava_unsupported;
}

CUresult
cuGetErrorName(CUresult error, const char** pStr)
{
    ava_argument(pStr) {
        ava_out; ava_buffer(1);
        ava_element {
            ava_lifetime_manual;
            ava_buffer(100);
        }
    }
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
    CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    ava_argument(numBlocks) {
        ava_out; ava_buffer(1);
    }

    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks,
    CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    ava_argument(numBlocks) {
        ava_out; ava_buffer(1);
    }

    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
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

CUBLASAPI cublasStatus_t  CUBLASWINAPI
cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t  CUBLASWINAPI
cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSetLoggerCallback(cublasLogCallback userCallback)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasGetLoggerCallback(cublasLogCallback* userCallback)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cublasStatus_t CUBLASWINAPI cublasSetVector (int n, int elemSize, const void *x,
                                             int incx, void *devicePtr, int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cublasStatus_t CUBLASWINAPI cublasGetVector (int n, int elemSize, const void *x,
                                             int incx, void *y, int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t  CUBLASWINAPI
cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t  CUBLASWINAPI
cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
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
        ava_opaque;
    }
}

cublasStatus_t CUBLASWINAPI
cublasGetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda,
                void *B, int ldb)
{
    ava_argument(A) {
        ava_opaque;
    }

    ava_argument(B) {
        ava_out; ava_buffer(rows * cols * elemSize);
    }
}

cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
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

/* ---------------- CUBLAS BLAS1 functions ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle,
                                                   int n,
                                                   const void *x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void *result,
                                                   cudaDataType resultType,
                                                   cudaDataType executionType) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float *x,
                                                     int incx,
                                                     float *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double *x,
                                                     int incx,
                                                     double *result)  /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      float *result)  /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      double *result)  /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx (cublasHandle_t handle,
                                                   int n,
                                                   const void *x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   const void *y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   void *result,
                                                   cudaDataType resultType,
                                                   cudaDataType executionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx (cublasHandle_t handle,
                                                    int n,
                                                    const void *x,
                                                    cudaDataType xType,
                                                    int incx,
                                                    const void *y,
                                                    cudaDataType yType,
                                                    int incy,
                                                    void *result,
                                                    cudaDataType resultType,
                                                    cudaDataType executionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2 (cublasHandle_t handle,
                                                     int n,
                                                     const float *x,
                                                     int incx,
                                                     const float *y,
                                                     int incy,
                                                     float *result)  /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2 (cublasHandle_t handle,
                                                     int n,
                                                     const double *x,
                                                     int incx,
                                                     const double *y,
                                                     int incy,
                                                     double *result)  /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *result)  /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle,
                                                   int n,
                                                   const void *alpha,  /* host or device pointer */
                                                   cudaDataType alphaType,
                                                   void *x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   cudaDataType executionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double *alpha,  /* host or device pointer */
                                                     double *x,
                                                     int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     cuComplex *x,
                                                     int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     cuDoubleComplex *x,
                                                     int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx (cublasHandle_t handle,
                                                    int n,
                                                    const void *alpha, /* host or device pointer */
                                                    cudaDataType alphaType,
                                                    const void *x,
                                                    cudaDataType xType,
                                                    int incx,
                                                    void *y,
                                                    cudaDataType yType,
                                                    int incy,
                                                    cudaDataType executiontype)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *x,
                                                      int incx,
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *x,
                                                      int incx,
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *x,
                                                      int incx,
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCopyEx (cublasHandle_t handle,
                                                    int n,
                                                    const void *x,
                                                    cudaDataType xType,
                                                    int incx,
                                                    void *y,
                                                    cudaDataType yType,
                                                    int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const float *x,
                                                      int incx,
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const double *x,
                                                      int incx,
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2 (cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2 (cublasHandle_t handle,
                                                      int n,
                                                      float *x,
                                                      int incx,
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2 (cublasHandle_t handle,
                                                      int n,
                                                      double *x,
                                                      int incx,
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2 (cublasHandle_t handle,
                                                      int n,
                                                      cuComplex *x,
                                                      int incx,
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2 (cublasHandle_t handle,
                                                      int n,
                                                      cuDoubleComplex *x,
                                                      int incx,
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSwapEx (cublasHandle_t handle,
                                                    int n,
                                                    void *x,
                                                    cudaDataType xType,
                                                    int incx,
                                                    void *y,
                                                    cudaDataType yType,
                                                    int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle,
                                                      int n,
                                                      const float *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle,
                                                      int n,
                                                      const double *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */

{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIamaxEx(cublasHandle_t handle,
                                                    int n,
                                                    const void *x, cudaDataType xType,
                                                    int incx,
                                                    int *result  /* host or device pointer */
                                                    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle,
                                                      int n,
                                                      const float *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle,
                                                      int n,
                                                      const double *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      int *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIaminEx(cublasHandle_t handle,
                                                      int n,
                                                      const void *x, cudaDataType xType,
                                                      int incx,
                                                      int *result /* host or device pointer */
                                                    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAsumEx(cublasHandle_t handle,
                                                   int n,
                                                   const void *x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void *result,
                                                   cudaDataType resultType, /* host or device pointer */
                                                   cudaDataType executiontype
                                                  )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float *x,
                                                     int incx,
                                                     float *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double *x,
                                                     int incx,
                                                     double *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incx,
                                                      float *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      double *result) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2 (cublasHandle_t handle,
                                                     int n,
                                                     float *x,
                                                     int incx,
                                                     float *y,
                                                     int incy,
                                                     const float *c,  /* host or device pointer */
                                                     const float *s) /* host or device pointer */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2 (cublasHandle_t handle,
                                                     int n,
                                                     double *x,
                                                     int incx,
                                                     double *y,
                                                     int incy,
                                                     const double *c,  /* host or device pointer */
                                                     const double *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2 (cublasHandle_t handle,
                                                     int n,
                                                     cuComplex *x,
                                                     int incx,
                                                     cuComplex *y,
                                                     int incy,
                                                     const float *c,      /* host or device pointer */
                                                     const cuComplex *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle,
                                                     int n,
                                                     cuComplex *x,
                                                     int incx,
                                                     cuComplex *y,
                                                     int incy,
                                                     const float *c,  /* host or device pointer */
                                                     const float *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2 (cublasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *y,
                                                     int incy,
                                                     const double *c,            /* host or device pointer */
                                                     const cuDoubleComplex *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *y,
                                                     int incy,
                                                     const double *c,  /* host or device pointer */
                                                     const double *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx (cublasHandle_t handle,
                                                     int n,
                                                     void *x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     void *y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     const void *c,  /* host or device pointer */
                                                     const void *s,
                                                     cudaDataType csType,
                                                     cudaDataType executiontype)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle,
                                                     float *a,   /* host or device pointer */
                                                     float *b,   /* host or device pointer */
                                                     float *c,   /* host or device pointer */
                                                     float *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle,
                                                     double *a,  /* host or device pointer */
                                                     double *b,  /* host or device pointer */
                                                     double *c,  /* host or device pointer */
                                                     double *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle,
                                                     cuComplex *a,  /* host or device pointer */
                                                     cuComplex *b,  /* host or device pointer */
                                                     float *c,      /* host or device pointer */
                                                     cuComplex *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle,
                                                     cuDoubleComplex *a,  /* host or device pointer */
                                                     cuDoubleComplex *b,  /* host or device pointer */
                                                     double *c,           /* host or device pointer */
                                                     cuDoubleComplex *s)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotgEx(cublasHandle_t handle,
                                                     void *a,   /* host or device pointer */
                                                     void *b,   /* host or device pointer */
                                                     cudaDataType abType,
                                                     void *c,   /* host or device pointer */
                                                     void *s,   /* host or device pointer */
                                                     cudaDataType csType,
                                                     cudaDataType executiontype)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle,
                                                     int n,
                                                     float *x,
                                                     int incx,
                                                     float *y,
                                                     int incy,
                                                     const float* param)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle,
                                                     int n,
                                                     double *x,
                                                     int incx,
                                                     double *y,
                                                     int incy,
                                                     const double* param)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmEx(cublasHandle_t handle,
                                                     int n,
                                                     void *x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     void *y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     const void* param, /* host or device pointer */
                                                     cudaDataType paramType,
                                                     cudaDataType executiontype)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle,
                                                      float *d1,        /* host or device pointer */
                                                      float *d2,        /* host or device pointer */
                                                      float *x1,        /* host or device pointer */
                                                      const float *y1,  /* host or device pointer */
                                                      float *param)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}    /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle,
                                                      double *d1,        /* host or device pointer */
                                                      double *d2,        /* host or device pointer */
                                                      double *x1,        /* host or device pointer */
                                                      const double *y1,  /* host or device pointer */
                                                      double *param)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}    /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmgEx(cublasHandle_t handle,
                                                    void *d1,        /* host or device pointer */
                                                    cudaDataType d1Type,
                                                    void *d2,        /* host or device pointer */
                                                    cudaDataType d2Type,
                                                    void *x1,        /* host or device pointer */
                                                    cudaDataType x1Type,
                                                    const void *y1,  /* host or device pointer */
                                                    cudaDataType y1Type,
                                                    void *param,     /* host or device pointer */
                                                    cudaDataType paramType,
                                                    cudaDataType executiontype
                                                    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta,  /* host or device pointer */
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* GBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TRMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *AP,
                                                      float *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *AP,
                                                      double *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *AP,
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *AP,
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TRSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TPSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *AP,
                                                      float *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *AP,
                                                      double *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *AP,
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *AP,
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* TBSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SYMV/HEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta,   /* host or device pointer */
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta,   /* host or device pointer */
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SBMV/HBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const float *alpha,   /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta,  /* host or device pointer */
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const double *alpha,   /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta,   /* host or device pointer */
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SPMV/HPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,  /* host or device pointer */
                                                      const float *AP,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta,   /* host or device pointer */
                                                      float *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *AP,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta,  /* host or device pointer */
                                                      double *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *AP,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *AP,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *y,
                                                      int incy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* GER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2 (cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *x,
                                                     int incx,
                                                     const float *y,
                                                     int incy,
                                                     float *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2 (cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *x,
                                                     int incx,
                                                     const double *y,
                                                     int incy,
                                                     double *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SYR/HER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *x,
                                                     int incx,
                                                     float *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *x,
                                                     int incx,
                                                     double *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *A,
                                                     int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SPR/HPR */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *x,
                                                     int incx,
                                                     float *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const double *x,
                                                     int incx,
                                                     double *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SYR2/HER2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *x,
                                                      int incx,
                                                      const double *y,
                                                      int incy,
                                                      double *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n,
                                                      const cuComplex *alpha,  /* host or device pointer */
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n,
                                                      const cuComplex *alpha,  /* host or device pointer */
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SPR2/HPR2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,  /* host or device pointer */
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,  /* host or device pointer */
                                                      const double *x,
                                                      int incx,
                                                      const double *y,
                                                      int incy,
                                                      double *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *AP)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
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
    ava_argument(A) ava_opaque;
    ava_argument(B) ava_opaque;
    ava_argument(C) ava_opaque;
    /* XXX I _think_ these are always device pointers for tensorflow ! */
    ava_argument(alpha) { ava_in; ava_buffer(1); }
    ava_argument(beta)  { ava_in; ava_buffer(1); }
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2 (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta, /* host or device pointer */
                                                      double *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2 (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3m  (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
 CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mEx (cublasHandle_t handle,
                                                     cublasOperation_t transa, cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const cuComplex *alpha,
                                                     const void *A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     const void *B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     const cuComplex *beta,
                                                     void *C,
                                                     cudaDataType Ctype,
                                                     int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2 (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm3m  (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm    (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const __half *alpha, /* host or device pointer */
                                                      const __half *A,
                                                      int lda,
                                                      const __half *B,
                                                      int ldb,
                                                      const __half *beta, /* host or device pointer */
                                                      __half *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
#endif
/* IO in FP16/FP32, computation in float */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmEx  (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      const float *beta, /* host or device pointer */
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx  (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const void *alpha, /* host or device pointer */
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      const void *beta, /* host or device pointer */
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      cudaDataType computeType,
                                                      cublasGemmAlgo_t algo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmEx (cublasHandle_t handle,
                                                     cublasOperation_t transa, cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const cuComplex *alpha,
                                                     const void *A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     const void *B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     const cuComplex *beta,
                                                     void *C,
                                                     cudaDataType Ctype,
                                                     int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasUint8gemmBias (cublasHandle_t handle,
                                                           cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc,
                                                           int m, int n, int k,
                                                           const unsigned char *A, int A_bias, int lda,
                                                           const unsigned char *B, int B_bias, int ldb,
                                                                 unsigned char *C, int C_bias, int ldc,
                                                           int C_mult, int C_shift)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* SYRK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *beta, /* host or device pointer */
                                                      float *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */
                                                      double *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkEx ( cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const cuComplex *beta,
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */
                                                      cuComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */
                                                      cuDoubleComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkEx  (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk3mEx (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const float *alpha,
                                                       const void *A, cudaDataType Atype,
                                                       int lda,
                                                       const float *beta,
                                                       void *C,
                                                       cudaDataType Ctype,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* SYR2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const float *alpha, /* host or device pointer */
                                                       const float *A,
                                                       int lda,
                                                       const float *B,
                                                       int ldb,
                                                       const float *beta, /* host or device pointer */
                                                       float *C,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const double *alpha, /* host or device pointer */
                                                       const double *A,
                                                       int lda,
                                                       const double *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */
                                                       double *C,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha, /* host or device pointer */
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const cuComplex *beta, /* host or device pointer */
                                                       cuComplex *C,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha,  /* host or device pointer */
                                                       const cuDoubleComplex *A,
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const cuDoubleComplex *beta,  /* host or device pointer */
                                                       cuDoubleComplex *C,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* HER2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha, /* host or device pointer */
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const float *beta,   /* host or device pointer */
                                                       cuComplex *C,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha, /* host or device pointer */
                                                       const cuDoubleComplex *A,
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */
                                                       cuDoubleComplex *C,
                                                       int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* SYRKX : eXtended SYRK*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float *alpha, /* host or device pointer */
                                                    const float *A,
                                                    int lda,
                                                    const float *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */
                                                    float *C,
                                                    int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const double *alpha, /* host or device pointer */
                                                    const double *A,
                                                    int lda,
                                                    const double *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */
                                                    double *C,
                                                    int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha, /* host or device pointer */
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const cuComplex *beta, /* host or device pointer */
                                                    cuComplex *C,
                                                    int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const cuDoubleComplex *beta, /* host or device pointer */
                                                    cuDoubleComplex *C,
                                                    int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* HERKX : eXtended HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha, /* host or device pointer */
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */
                                                    cuComplex *C,
                                                    int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */
                                                    cuDoubleComplex *C,
                                                    int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* SYMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      const float *beta, /* host or device pointer */
                                                      float *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta, /* host or device pointer */
                                                      double *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* HEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */
                                                      cuComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */
                                                      cuDoubleComplex *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TRSM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      float *B,
                                                      int ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      double *B,
                                                      int ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *B,
                                                     int ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     cuDoubleComplex *B,
                                                     int ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

 /* TRMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      float *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      double *C,
                                                      int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     const cuComplex *A,
                                                     int lda,
                                                     const cuComplex *B,
                                                     int ldb,
                                                     cuComplex *C,
                                                     int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     const cuDoubleComplex *B,
                                                     int ldb,
                                                     cuDoubleComplex *C,
                                                     int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* BATCH GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const float *alpha,  /* host or device pointer */
                                                          const float *const Aarray[],
                                                          int lda,
                                                          const float *const Barray[],
                                                          int ldb,
                                                          const float *beta,   /* host or device pointer */
                                                          float *const Carray[],
                                                          int ldc,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const double *alpha,  /* host or device pointer */
                                                          const double *const Aarray[],
                                                          int lda,
                                                          const double *const Barray[],
                                                          int ldb,
                                                          const double *beta,  /* host or device pointer */
                                                          double *const Carray[],
                                                          int ldc,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha, /* host or device pointer */
                                                          const cuComplex *const Aarray[],
                                                          int lda,
                                                          const cuComplex *const Barray[],
                                                          int ldb,
                                                          const cuComplex *beta, /* host or device pointer */
                                                          cuComplex *const Carray[],
                                                          int ldc,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha, /* host or device pointer */
                                                          const cuComplex *const Aarray[],
                                                          int lda,
                                                          const cuComplex *const Barray[],
                                                          int ldb,
                                                          const cuComplex *beta, /* host or device pointer */
                                                          cuComplex *const Carray[],
                                                          int ldc,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuDoubleComplex *alpha, /* host or device pointer */
                                                          const cuDoubleComplex *const Aarray[],
                                                          int lda,
                                                          const cuDoubleComplex *const Barray[],
                                                          int ldb,
                                                          const cuDoubleComplex *beta, /* host or device pointer */
                                                          cuDoubleComplex *const Carray[],
                                                          int ldc,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

ava_utility int __helper_a_last_dim_size(cublasOperation_t transa, int k, int m)
{
    if (transa == CUBLAS_OP_N) {
        return k;
    } else {
        return m;
    }
}

ava_utility int __helper_b_last_dim_size(cublasOperation_t transb, int k, int n)
{
    if (transb == CUBLAS_OP_N) {
        return n;
    } else {
        return k;
    }
}

ava_utility int __helper_type_size(cudaDataType dataType)
{
    switch (dataType) {
        case CUDA_R_16F: return 2;
        case CUDA_C_16F: return 4;
        case CUDA_R_32F: return 4;
        case CUDA_C_32F: return sizeof(float _Complex);
        case CUDA_R_64F: return 8;
        case CUDA_C_64F: return sizeof(double _Complex);
        case CUDA_R_8I: return 1;
        case CUDA_C_8I: return 2;
        case CUDA_R_8U: return 1;
        case CUDA_C_8U: return 2;
        case CUDA_R_32I: return 4;
        case CUDA_C_32I: return 8;
        case CUDA_R_32U: return 4;
        case CUDA_C_32U: return 8;
        default: fprintf(stderr, "invalid data type: %d\n", dataType);
                 abort();
    }
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const void *alpha, /* host or device pointer */
                                                          const void *const Aarray[],
                                                          cudaDataType Atype,
                                                          int lda,
                                                          const void *const Barray[],
                                                          cudaDataType Btype,
                                                          int ldb,
                                                          const void *beta, /* host or device pointer */
                                                          void *const Carray[],
                                                          cudaDataType Ctype,
                                                          int ldc,
                                                          int batchCount,
                                                          cudaDataType computeType,
                                                          cublasGemmAlgo_t algo)
{
    ava_argument(handle) ava_handle;
    // In tensorflow, Aarray, Barray and Carray are device memory
    ava_argument(Aarray) ava_opaque;
    ava_argument(Barray) ava_opaque;
    ava_argument(Carray) ava_opaque;
    // If they are host memory, use the following code:
    /*
    ava_argument(Aarray) {
        ava_in; ava_buffer(batchCount);
        ava_element {
            ava_in; ava_buffer(lda * __helper_a_last_dim_size(transa, k, m) * __helper_type_size(Atype));
        }
    }

    ava_argument(Barray) {
        ava_in; ava_buffer(batchCount);
        ava_element {
            ava_in; ava_buffer(ldb * __helper_b_last_dim_size(transb, k, n) * __helper_type_size(Btype));
        }
    }

    ava_argument(Carray) {
        ava_type_cast(void**);
        ava_out; ava_buffer(batchCount);
        ava_element {
            ava_out; ava_buffer(ldc * n * __helper_type_size(Ctype));
        }
    }
    */
    // TODO: figure out alpha and beta
    ava_argument(alpha) { ava_in; ava_buffer(1); }
    ava_argument(beta)  { ava_in; ava_buffer(1); }
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx (cublasHandle_t handle,
                                                                  cublasOperation_t transa,
                                                                  cublasOperation_t transb,
                                                                  int m,
                                                                  int n,
                                                                  int k,
                                                                  const void *alpha,  /* host or device pointer */
                                                                  const void *A,
                                                                  cudaDataType Atype,
                                                                  int lda,
                                                                  long long int strideA,   /* purposely signed */
                                                                  const void *B,
                                                                  cudaDataType Btype,
                                                                  int ldb,
                                                                  long long int strideB,
                                                                  const void *beta,   /* host or device pointer */
                                                                  void *C,
                                                                  cudaDataType Ctype,
                                                                  int ldc,
                                                                  long long int strideC,
                                                                  int batchCount,
                                                                  cudaDataType computeType,
                                                                  cublasGemmAlgo_t algo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const float *alpha,  /* host or device pointer */
                                                                 const float *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const float *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const float *beta,   /* host or device pointer */
                                                                 float *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const double *alpha,  /* host or device pointer */
                                                                 const double *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const double *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const double *beta,   /* host or device pointer */
                                                                 double *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuComplex *alpha,  /* host or device pointer */
                                                                 const cuComplex *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const cuComplex *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const cuComplex *beta,   /* host or device pointer */
                                                                 cuComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuComplex *alpha,  /* host or device pointer */
                                                                 const cuComplex *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const cuComplex *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const cuComplex *beta,   /* host or device pointer */
                                                                 cuComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuDoubleComplex *alpha,  /* host or device pointer */
                                                                 const cuDoubleComplex *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const cuDoubleComplex *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const cuDoubleComplex *beta,   /* host or device poi */
                                                                 cuDoubleComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const float *alpha, /* host or device pointer */
                                                  const float *A,
                                                  int lda,
                                                  const float *beta , /* host or device pointer */
                                                  const float *B,
                                                  int ldb,
                                                  float *C,
                                                  int ldc)
{
    ava_argument(handle) ava_handle;
    ava_argument(alpha) { ava_in; ava_buffer(1); }
    ava_argument(A) ava_opaque;
    ava_argument(beta)  { ava_in; ava_buffer(1); }
    ava_argument(B) ava_opaque;
    ava_argument(C) ava_opaque;
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const double *alpha, /* host or device pointer */
                                                  const double *A,
                                                  int lda,
                                                  const double *beta, /* host or device pointer */
                                                  const double *B,
                                                  int ldb,
                                                  double *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuComplex *alpha, /* host or device pointer */
                                                  const cuComplex *A,
                                                  int lda,
                                                  const cuComplex *beta, /* host or device pointer */
                                                  const cuComplex *B,
                                                  int ldb,
                                                  cuComplex *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex *alpha, /* host or device pointer */
                                                  const cuDoubleComplex *A,
                                                  int lda,
                                                  const cuDoubleComplex *beta, /* host or device pointer */
                                                  const cuDoubleComplex *B,
                                                  int ldb,
                                                  cuDoubleComplex *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Batched LU - GETRF*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  float *const A[],                /*Device pointer*/
                                                  int lda,
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  double *const A[],               /*Device pointer*/
                                                  int lda,
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  cuComplex *const A[],           /*Device pointer*/
                                                  int lda,
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  cuDoubleComplex *const A[],     /*Device pointer*/
                                                  int lda,
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Batched inversion based on LU factorization from getrf */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const float *const A[],         /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  float *const C[],               /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const double *const A[],        /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  double *const C[],              /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuComplex *const A[],     /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  cuComplex *const C[],           /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuDoubleComplex *const A[], /*Device pointer*/
                                                  int lda,
                                                  const int *P,                     /*Device pointer*/
                                                  cuDoubleComplex *const C[],       /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Batched solver based on LU factorization from getrf */

CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasSgetrsBatched( cublasHandle_t handle,
                                                            cublasOperation_t trans,
                                                            int n,
                                                            int nrhs,
                                                            const float *const Aarray[],
                                                            int lda,
                                                            const int *devIpiv,
                                                            float *const Barray[],
                                                            int ldb,
                                                            int *info,
                                                            int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int n,
                                                           int nrhs,
                                                           const double *const Aarray[],
                                                           int lda,
                                                           const int *devIpiv,
                                                           double *const Barray[],
                                                           int ldb,
                                                           int *info,
                                                           int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasCgetrsBatched( cublasHandle_t handle,
                                                            cublasOperation_t trans,
                                                            int n,
                                                            int nrhs,
                                                            const cuComplex *const Aarray[],
                                                            int lda,
                                                            const int *devIpiv,
                                                            cuComplex *const Barray[],
                                                            int ldb,
                                                            int *info,
                                                            int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasZgetrsBatched( cublasHandle_t handle,
                                                            cublasOperation_t trans,
                                                            int n,
                                                            int nrhs,
                                                            const cuDoubleComplex *const Aarray[],
                                                            int lda,
                                                            const int *devIpiv,
                                                            cuDoubleComplex *const Barray[],
                                                            int ldb,
                                                            int *info,
                                                            int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* TRSM - Batched Triangular Solver */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched( cublasHandle_t    handle,
                                                          cublasSideMode_t  side,
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans,
                                                          cublasDiagType_t  diag,
                                                          int m,
                                                          int n,
                                                          const float *alpha,           /*Host or Device Pointer*/
                                                          const float *const A[],
                                                          int lda,
                                                          float *const B[],
                                                          int ldb,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched( cublasHandle_t    handle,
                                                          cublasSideMode_t  side,
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans,
                                                          cublasDiagType_t  diag,
                                                          int m,
                                                          int n,
                                                          const double *alpha,          /*Host or Device Pointer*/
                                                          const double *const A[],
                                                          int lda,
                                                          double *const B[],
                                                          int ldb,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched( cublasHandle_t    handle,
                                                          cublasSideMode_t  side,
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans,
                                                          cublasDiagType_t  diag,
                                                          int m,
                                                          int n,
                                                          const cuComplex *alpha,       /*Host or Device Pointer*/
                                                          const cuComplex *const A[],
                                                          int lda,
                                                          cuComplex *const B[],
                                                          int ldb,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched( cublasHandle_t    handle,
                                                          cublasSideMode_t  side,
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans,
                                                          cublasDiagType_t  diag,
                                                          int m,
                                                          int n,
                                                          const cuDoubleComplex *alpha, /*Host or Device Pointer*/
                                                          const cuDoubleComplex *const A[],
                                                          int lda,
                                                          cuDoubleComplex *const B[],
                                                          int ldb,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Batched - MATINV*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const float *const A[],      /*Device pointer*/
                                                          int lda,
                                                          float *const Ainv[],         /*Device pointer*/
                                                          int lda_inv,
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const double *const A[],     /*Device pointer*/
                                                          int lda,
                                                          double *const Ainv[],        /*Device pointer*/
                                                          int lda_inv,
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuComplex *const A[],  /*Device pointer*/
                                                          int lda,
                                                          cuComplex *const Ainv[],     /*Device pointer*/
                                                          int lda_inv,
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuDoubleComplex *const A[], /*Device pointer*/
                                                          int lda,
                                                          cuDoubleComplex *const Ainv[],    /*Device pointer*/
                                                          int lda_inv,
                                                          int *info,                        /*Device Pointer*/
                                                          int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Batch QR Factorization */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched( cublasHandle_t handle,
                                                           int m,
                                                           int n,
                                                           float *const Aarray[],      /*Device pointer*/
                                                           int lda,
                                                           float *const TauArray[],    /*Device pointer*/
                                                           int *info,
                                                           int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasDgeqrfBatched( cublasHandle_t handle,
                                                            int m,
                                                            int n,
                                                            double *const Aarray[],     /*Device pointer*/
                                                            int lda,
                                                            double *const TauArray[],   /*Device pointer*/
                                                            int *info,
                                                            int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasCgeqrfBatched( cublasHandle_t handle,
                                                            int m,
                                                            int n,
                                                            cuComplex *const Aarray[],          /*Device pointer*/
                                                            int lda,
                                                            cuComplex *const TauArray[],        /*Device pointer*/
                                                            int *info,
                                                            int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasZgeqrfBatched( cublasHandle_t handle,
                                                            int m,
                                                            int n,
                                                            cuDoubleComplex *const Aarray[],    /*Device pointer*/
                                                            int lda,
                                                            cuDoubleComplex *const TauArray[],  /*Device pointer*/
                                                            int *info,
                                                            int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* Least Square Min only m >= n and Non-transpose supported */
CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasSgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           float *const Aarray[],      /*Device pointer*/
                                                           int lda,
                                                           float *const Carray[],      /*Device pointer*/
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,          /*Device pointer*/
                                                           int batchSize )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasDgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           double *const Aarray[],     /*Device pointer*/
                                                           int lda,
                                                           double *const Carray[],     /*Device pointer*/
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,          /*Device pointer*/
                                                           int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasCgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           cuComplex *const Aarray[],  /*Device pointer*/
                                                           int lda,
                                                           cuComplex *const Carray[],  /*Device pointer*/
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasZgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           cuDoubleComplex *const Aarray[],  /*Device pointer*/
                                                           int lda,
                                                           cuDoubleComplex *const Carray[],  /*Device pointer*/
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/* DGMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const float *A,
                                                  int lda,
                                                  const float *x,
                                                  int incx,
                                                  float *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const double *A,
                                                  int lda,
                                                  const double *x,
                                                  int incx,
                                                  double *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuComplex *A,
                                                  int lda,
                                                  const cuComplex *x,
                                                  int incx,
                                                  cuComplex *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex *A,
                                                  int lda,
                                                  const cuDoubleComplex *x,
                                                  int incx,
                                                  cuDoubleComplex *C,
                                                  int ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* TPTTR : Triangular Pack format to Triangular format */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *AP,
                                                     float *A,
                                                     int lda )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *AP,
                                                     double *A,
                                                     int lda )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *AP,
                                                     cuComplex *A,
                                                     int lda )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *AP,
                                                     cuDoubleComplex *A,
                                                     int lda )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
 /* TRTTP : Triangular format to Triangular Pack format */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *A,
                                                     int lda,
                                                     float *AP )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *A,
                                                     int lda,
                                                     double *AP )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *AP )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     cuDoubleComplex *AP )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
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
   ava_argument(x) ava_opaque;
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_opaque;
   ava_argument(bnScaleBiasMeanVarDesc) ava_handle;
   ava_argument(bnScale) ava_opaque;
   ava_argument(bnBias) ava_opaque;
   ava_argument(estimatedMean) ava_opaque;
   ava_argument(estimatedVariance) ava_opaque;
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
   ava_argument(x) ava_opaque;
   ava_argument(wDesc) ava_handle;
   ava_argument(w) ava_opaque;
   ava_argument(convDesc) ava_handle;
   ava_argument(workSpace) ava_opaque;
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_opaque;
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
cudnnDestroy(cudnnHandle_t handle)
{
   ava_argument(handle) ava_handle;
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
cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t                      handle,
                                    const cudnnTensorDescriptor_t      xDesc,
                                    const cudnnFilterDescriptor_t      wDesc,
                                    const cudnnConvolutionDescriptor_t convDesc,
                                    const cudnnTensorDescriptor_t      yDesc,
                                    cudnnConvolutionFwdPreference_t    preference,
                                    size_t                             memoryLimitInBytes,
                                    cudnnConvolutionFwdAlgo_t         *algo)
{
    ava_argument(handle) ava_handle;
    ava_argument(xDesc) ava_handle;
    ava_argument(wDesc) ava_handle;
    ava_argument(convDesc) ava_handle;
    ava_argument(yDesc) ava_handle;
    ava_argument(algo) {
        ava_out; ava_buffer(1);
    }
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t                       handle,
                                       const cudnnTensorDescriptor_t       xDesc,
                                       const cudnnFilterDescriptor_t       wDesc,
                                       const cudnnConvolutionDescriptor_t  convDesc,
                                       const cudnnTensorDescriptor_t       yDesc,
                                       const int                           requestedAlgoCount,
                                       int                                *returnedAlgoCount,
                                       cudnnConvolutionFwdAlgoPerf_t      *perfResults)
{
    ava_argument(handle) ava_handle;
    ava_argument(xDesc) ava_handle;
    ava_argument(wDesc) ava_handle;
    ava_argument(convDesc) ava_handle;
    ava_argument(yDesc) ava_handle;
    ava_argument(returnedAlgoCount) {
        ava_out; ava_buffer(1);
    }
    ava_argument(perfResults) {
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
   ava_argument(x) ava_opaque;
   ava_argument(beta) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(yDesc) ava_handle;
   ava_argument(y) ava_opaque;
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
   ava_argument(y) ava_opaque;
   ava_argument(dyDesc) ava_handle;
   ava_argument(dy) ava_opaque;
   ava_argument(xDesc) ava_handle;
   ava_argument(x) ava_opaque;
   ava_argument(beta) {
      ava_type_cast(const double *);
      ava_in; ava_buffer(1);
   }
   ava_argument(dxDesc) ava_handle;
   ava_argument(dx) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc)
{
    ava_argument(rnnDesc) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc)
{
   ava_argument(rnnDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDescriptor(cudnnHandle_t handle,
                      cudnnRNNDescriptor_t rnnDesc,
                      const int hiddenSize,
                      const int numLayers,
                      cudnnDropoutDescriptor_t dropoutDesc,
                      cudnnRNNInputMode_t inputMode,
                      cudnnDirectionMode_t direction,
                      cudnnRNNMode_t mode,
                      cudnnRNNAlgo_t algo,
                      cudnnDataType_t mathPrec)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNDescriptor(cudnnHandle_t handle,
                      cudnnRNNDescriptor_t rnnDesc,
                      int *hiddenSize,
                      int *numLayers,
                      cudnnDropoutDescriptor_t *dropoutDesc,
                      cudnnRNNInputMode_t *inputMode,
                      cudnnDirectionMode_t *direction,
                      cudnnRNNMode_t *mode,
                      cudnnRNNAlgo_t *algo,
                      cudnnDataType_t *mathPrec)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNSetClip(cudnnHandle_t handle,
                cudnnRNNDescriptor_t rnnDesc,
                cudnnRNNClipMode_t clipMode,
                cudnnNanPropagation_t clipNanOpt,
                double lclip,
                double rclip)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNGetClip(cudnnHandle_t handle,
                cudnnRNNDescriptor_t rnnDesc,
                cudnnRNNClipMode_t *clipMode,
                cudnnNanPropagation_t *clipNanOpt,
                double *lclip,
                double *rclip)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                            cudnnRNNDescriptor_t rnnDesc,
                            const int recProjSize,
                            const int outProjSize)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                            const cudnnRNNDescriptor_t rnnDesc,
                            int *recProjSize,
                            int *outProjSize)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

/* Expensive. Creates the plan for the specific settings. */
cudnnStatus_t CUDNNWINAPI
cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                             const int minibatch,
                             const cudnnDataType_t dataType,
                             cudnnPersistentRNNPlan_t *plan)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

/* dataType in weight descriptors and input descriptors is used to describe storage */
cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                         const cudnnRNNDescriptor_t rnnDesc,
                         const int seqLength,
                         const cudnnTensorDescriptor_t *xDesc,
                         size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
                               const cudnnRNNDescriptor_t rnnDesc,
                               const int seqLength,
                               const cudnnTensorDescriptor_t *xDesc,
                               size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNParamsSize(cudnnHandle_t handle,
                      const cudnnRNNDescriptor_t rnnDesc,
                      const cudnnTensorDescriptor_t xDesc,
                      size_t *sizeInBytes,
                      cudnnDataType_t dataType)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
                                const cudnnRNNDescriptor_t rnnDesc,
                                const int pseudoLayer,
                                const cudnnTensorDescriptor_t xDesc,
                                const cudnnFilterDescriptor_t wDesc,
                                const void *w,
                                const int linLayerID,
                                cudnnFilterDescriptor_t linLayerMatDesc,
                                void **linLayerMat)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
                              const cudnnRNNDescriptor_t rnnDesc,
                              const int pseudoLayer,
                              const cudnnTensorDescriptor_t xDesc,
                              const cudnnFilterDescriptor_t wDesc,
                              const void *w,
                              const int linLayerID,
                              cudnnFilterDescriptor_t linLayerBiasDesc,
                              void **linLayerBias)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardInference(cudnnHandle_t handle,
                         const cudnnRNNDescriptor_t rnnDesc,
                         const int seqLength,
                         const cudnnTensorDescriptor_t *xDesc,
                         const void *x,
                         const cudnnTensorDescriptor_t hxDesc,
                         const void *hx,
                         const cudnnTensorDescriptor_t cxDesc,
                         const void *cx,
                         const cudnnFilterDescriptor_t wDesc,
                         const void *w,
                         const cudnnTensorDescriptor_t *yDesc,
                         void *y,
                         const cudnnTensorDescriptor_t hyDesc,
                         void *hy,
                         const cudnnTensorDescriptor_t cyDesc,
                         void *cy,
                         void *workspace,
                         size_t workSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardTraining(cudnnHandle_t handle,
                        const cudnnRNNDescriptor_t rnnDesc,
                        const int seqLength,
                        const cudnnTensorDescriptor_t *xDesc,
                        const void *x,
                        const cudnnTensorDescriptor_t hxDesc,
                        const void *hx,
                        const cudnnTensorDescriptor_t cxDesc,
                        const void *cx,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnTensorDescriptor_t *yDesc,
                        void *y,
                        const cudnnTensorDescriptor_t hyDesc,
                        void *hy,
                        const cudnnTensorDescriptor_t cyDesc,
                        void *cy,
                        void *workspace,
                        size_t workSpaceSizeInBytes,
                        void *reserveSpace,
                        size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardData(cudnnHandle_t handle,
                     const cudnnRNNDescriptor_t rnnDesc,
                     const int seqLength,
                     const cudnnTensorDescriptor_t *yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t *dyDesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t dhyDesc,
                     const void *dhy,
                     const cudnnTensorDescriptor_t dcyDesc,
                     const void *dcy,
                     const cudnnFilterDescriptor_t wDesc,
                     const void *w,
                     const cudnnTensorDescriptor_t hxDesc,
                     const void *hx,
                     const cudnnTensorDescriptor_t cxDesc,
                     const void *cx,
                     const cudnnTensorDescriptor_t *dxDesc,
                     void *dx,
                     const cudnnTensorDescriptor_t dhxDesc,
                     void *dhx,
                     const cudnnTensorDescriptor_t dcxDesc,
                     void *dcx,
                     void *workspace,
                     size_t workSpaceSizeInBytes,
                     void *reserveSpace,
                     size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardWeights(cudnnHandle_t handle,
                        const cudnnRNNDescriptor_t rnnDesc,
                        const int seqLength,
                        const cudnnTensorDescriptor_t *xDesc,
                        const void *x,
                        const cudnnTensorDescriptor_t hxDesc,
                        const void *hx,
                        const cudnnTensorDescriptor_t *yDesc,
                        const void *y,
                        const void *workspace,
                        size_t workSpaceSizeInBytes,
                        const cudnnFilterDescriptor_t dwDesc,
                        void *dw,
                        const void *reserveSpace,
                        size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

/* RNN EX API */

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t dataType,
                          cudnnRNNDataLayout_t layout,
                          int maxSeqLength,
                          int batchSize,
                          int vectorSize,
                          const int seqLengthArray[], /* length of each sequence in the batch */
                          void *paddingFill)          /* symbol for filling padding position in output */
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t *dataType,
                          cudnnRNNDataLayout_t *layout,
                          int *maxSeqLength,
                          int *batchSize,
                          int *vectorSize,
                          int arrayLengthRequested,
                          int seqLengthArray[],
                          void *paddingFill)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardTrainingEx(cudnnHandle_t handle,
                          const cudnnRNNDescriptor_t rnnDesc,
                          const cudnnRNNDataDescriptor_t xDesc,
                          const void *x,
                          const cudnnTensorDescriptor_t hxDesc,
                          const void *hx,
                          const cudnnTensorDescriptor_t cxDesc,
                          const void *cx,
                          const cudnnFilterDescriptor_t wDesc,
                          const void *w,
                          const cudnnRNNDataDescriptor_t yDesc,
                          void *y,
                          const cudnnTensorDescriptor_t hyDesc,
                          void *hy,
                          const cudnnTensorDescriptor_t cyDesc,
                          void *cy,
                          const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
                          const void *keys,                     /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
                          void *cAttn,                          /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
                          void *iAttn,                          /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
                          void *queries,                        /* reserved, should pass NULL */
                          void *workSpace,
                          size_t workSpaceSizeInBytes,
                          void *reserveSpace,
                          size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardInferenceEx(cudnnHandle_t handle,
                           const cudnnRNNDescriptor_t rnnDesc,
                           const cudnnRNNDataDescriptor_t xDesc,
                           const void *x,
                           const cudnnTensorDescriptor_t hxDesc,
                           const void *hx,
                           const cudnnTensorDescriptor_t cxDesc,
                           const void *cx,
                           const cudnnFilterDescriptor_t wDesc,
                           const void *w,
                           const cudnnRNNDataDescriptor_t yDesc,
                           void *y,
                           const cudnnTensorDescriptor_t hyDesc,
                           void *hy,
                           const cudnnTensorDescriptor_t cyDesc,
                           void *cy,
                           const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
                           const void *keys,                     /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
                           void *cAttn,                          /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
                           void *iAttn,                          /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
                           void *queries,                        /* reserved, should pass NULL */
                           void *workSpace,
                           size_t workSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardDataEx(cudnnHandle_t handle,
                       const cudnnRNNDescriptor_t rnnDesc,
                       const cudnnRNNDataDescriptor_t yDesc,
                       const void *y,
                       const cudnnRNNDataDescriptor_t dyDesc,
                       const void *dy,
                       const cudnnRNNDataDescriptor_t dcDesc, /* reserved, should pass NULL */
                       const void *dcAttn,                    /* reserved, should pass NULL */
                       const cudnnTensorDescriptor_t dhyDesc,
                       const void *dhy,
                       const cudnnTensorDescriptor_t dcyDesc,
                       const void *dcy,
                       const cudnnFilterDescriptor_t wDesc,
                       const void *w,
                       const cudnnTensorDescriptor_t hxDesc,
                       const void *hx,
                       const cudnnTensorDescriptor_t cxDesc,
                       const void *cx,
                       const cudnnRNNDataDescriptor_t dxDesc,
                       void *dx,
                       const cudnnTensorDescriptor_t dhxDesc,
                       void *dhx,
                       const cudnnTensorDescriptor_t dcxDesc,
                       void *dcx,
                       const cudnnRNNDataDescriptor_t dkDesc, /* reserved, should pass NULL */
                       void *dkeys,                           /* reserved, should pass NULL */
                       void *workSpace,
                       size_t workSpaceSizeInBytes,
                       void *reserveSpace,
                       size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardWeightsEx(cudnnHandle_t handle,
                          const cudnnRNNDescriptor_t rnnDesc,
                          const cudnnRNNDataDescriptor_t xDesc,
                          const void *x,
                          const cudnnTensorDescriptor_t hxDesc,
                          const void *hx,
                          const cudnnRNNDataDescriptor_t yDesc,
                          const void *y,
                          void *workSpace,
                          size_t workSpaceSizeInBytes,
                          const cudnnFilterDescriptor_t dwDesc,
                          void *dw,
                          void *reserveSpace,
                          size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

/* RNN FIND API */

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
                                        const cudnnRNNDescriptor_t rnnDesc,
                                        const int seqLength,
                                        const cudnnTensorDescriptor_t *xDesc,
                                        const void *x,
                                        const cudnnTensorDescriptor_t hxDesc,
                                        const void *hx,
                                        const cudnnTensorDescriptor_t cxDesc,
                                        const void *cx,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const void *w,
                                        const cudnnTensorDescriptor_t *yDesc,
                                        void *y,
                                        const cudnnTensorDescriptor_t hyDesc,
                                        void *hy,
                                        const cudnnTensorDescriptor_t cyDesc,
                                        void *cy,
                                        const float findIntensity,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        cudnnAlgorithmPerformance_t *perfResults,
                                        void *workspace,
                                        size_t workSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnnDesc,
                                       const int seqLength,
                                       const cudnnTensorDescriptor_t *xDesc,
                                       const void *x,
                                       const cudnnTensorDescriptor_t hxDesc,
                                       const void *hx,
                                       const cudnnTensorDescriptor_t cxDesc,
                                       const void *cx,
                                       const cudnnFilterDescriptor_t wDesc,
                                       const void *w,
                                       const cudnnTensorDescriptor_t *yDesc,
                                       void *y,
                                       const cudnnTensorDescriptor_t hyDesc,
                                       void *hy,
                                       const cudnnTensorDescriptor_t cyDesc,
                                       void *cy,
                                       const float findIntensity,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnAlgorithmPerformance_t *perfResults,
                                       void *workspace,
                                       size_t workSpaceSizeInBytes,
                                       void *reserveSpace,
                                       size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t rnnDesc,
                                    const int seqLength,
                                    const cudnnTensorDescriptor_t *yDesc,
                                    const void *y,
                                    const cudnnTensorDescriptor_t *dyDesc,
                                    const void *dy,
                                    const cudnnTensorDescriptor_t dhyDesc,
                                    const void *dhy,
                                    const cudnnTensorDescriptor_t dcyDesc,
                                    const void *dcy,
                                    const cudnnFilterDescriptor_t wDesc,
                                    const void *w,
                                    const cudnnTensorDescriptor_t hxDesc,
                                    const void *hx,
                                    const cudnnTensorDescriptor_t cxDesc,
                                    const void *cx,
                                    const cudnnTensorDescriptor_t *dxDesc,
                                    void *dx,
                                    const cudnnTensorDescriptor_t dhxDesc,
                                    void *dhx,
                                    const cudnnTensorDescriptor_t dcxDesc,
                                    void *dcx,
                                    const float findIntensity,
                                    const int requestedAlgoCount,
                                    int *returnedAlgoCount,
                                    cudnnAlgorithmPerformance_t *perfResults,
                                    void *workspace,
                                    size_t workSpaceSizeInBytes,
                                    void *reserveSpace,
                                    size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnnDesc,
                                       const int seqLength,
                                       const cudnnTensorDescriptor_t *xDesc,
                                       const void *x,
                                       const cudnnTensorDescriptor_t hxDesc,
                                       const void *hx,
                                       const cudnnTensorDescriptor_t *yDesc,
                                       const void *y,
                                       const float findIntensity,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnAlgorithmPerformance_t *perfResults,
                                       const void *workspace,
                                       size_t workSpaceSizeInBytes,
                                       const cudnnFilterDescriptor_t dwDesc,
                                       void *dw,
                                       const void *reserveSpace,
                                       size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

/* DEPRECATED routines to be removed next release :
   User should use the non-suffixed version (which has the API and functionality of _v6 version)
   Routines with _v5 suffix has the functionality of the non-suffixed routines in the CUDNN V6
 */

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
                         cudnnRNNDescriptor_t rnnDesc,
                         const int hiddenSize,
                         const int numLayers,
                         cudnnDropoutDescriptor_t dropoutDesc,
                         cudnnRNNInputMode_t inputMode,
                         cudnnDirectionMode_t direction,
                         cudnnRNNMode_t mode,
                         cudnnRNNAlgo_t algo,
                         cudnnDataType_t mathPrec)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,
                         int hiddenSize,
                         int numLayers,
                         cudnnDropoutDescriptor_t dropoutDesc,
                         cudnnRNNInputMode_t inputMode,
                         cudnnDirectionMode_t direction,
                         cudnnRNNMode_t mode,
                         cudnnDataType_t mathPrec)
{
    fprintf(stderr, "%s is not implemented\n", __PRETTY_FUNCTION__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnTensorFormat_t format,
                           cudnnDataType_t dataType, /* image data type */
                           int n,                    /* number of inputs (batch size) */
                           int c,                    /* number of input feature maps */
                           int h,                    /* height of input section */
                           int w)                    /* width of input section */
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnDataType_t dataType, /* image data type */
                             int n,                    /* number of inputs (batch size) */
                             int c,                    /* number of input feature maps */
                             int h,                    /* height of input section */
                             int w,                    /* width of input section */
                             int nStride,
                             int cStride,
                             int hStride,
                             int wStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t *dataType, /* image data type */
                           int *n,                    /* number of inputs (batch size) */
                           int *c,                    /* number of input feature maps  */
                           int *h,                    /* height of input section */
                           int *w,                    /* width of input section */
                           int *nStride,
                           int *cStride,
                           int *hStride,
                           int *wStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested,
                           cudnnDataType_t *dataType,
                           int *nbDims,
                           int dimA[],
                           int strideA[])
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/** Create a destination descriptor for cudnnTransformTensor */
cudnnStatus_t CUDNNWINAPI
cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                       const cudnnTensorDescriptor_t srcDesc,
                       cudnnTensorDescriptor_t destDesc,
                       size_t *destSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/** Create an empty tensor transform descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/** Initialize a previously created tensor transform descriptor. */
cudnnStatus_t CUDNNWINAPI
cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  const uint32_t nbDims,
                                  const cudnnTensorFormat_t destFormat,
                                  const int32_t padBeforeA[],
                                  const int32_t padAfterA[],
                                  const uint32_t foldA[],
                                  const cudnnFoldingDirection_t direction)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/**
 * Retrieves the values stored in a previously initialized tensor transform
 * descriptor.
 */
cudnnStatus_t CUDNNWINAPI
cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  uint32_t nbDimsRequested,
                                  cudnnTensorFormat_t *destFormat,
                                  int32_t padBeforeA[],
                                  int32_t padAfterA[],
                                  uint32_t foldA[],
                                  cudnnFoldingDirection_t *direction)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/**
 * Destroys a previously created tensor transform descriptor.
 */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t CUDNNWINAPI
cudnnTransformTensor(cudnnHandle_t handle,
                     const void *alpha,
                     const cudnnTensorDescriptor_t xDesc,
                     const void *x,
                     const void *beta,
                     const cudnnTensorDescriptor_t yDesc,
                     void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnTransformTensorEx(cudnnHandle_t handle,
                       const cudnnTensorTransformDescriptor_t transDesc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t srcDesc,
                       const void *srcData,
                       const void *beta,
                       const cudnnTensorDescriptor_t destDesc,
                       void *destData)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Helper function to calculate folding descriptors  for dgrad */
cudnnStatus_t CUDNNWINAPI
cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,
                                          const cudnnFilterDescriptor_t filterDesc,
                                          const cudnnTensorDescriptor_t diffDesc,
                                          const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t gradDesc,
                                          const cudnnTensorFormat_t transformFormat,
                                          cudnnFilterDescriptor_t foldedFilterDesc,
                                          cudnnTensorDescriptor_t paddedDiffDesc,
                                          cudnnConvolutionDescriptor_t foldedConvDesc,
                                          cudnnTensorDescriptor_t foldedGradDesc,
                                          cudnnTensorTransformDescriptor_t filterFoldTransDesc,
                                          cudnnTensorTransformDescriptor_t diffPadTransDesc,
                                          cudnnTensorTransformDescriptor_t gradFoldTransDesc,
                                          cudnnTensorTransformDescriptor_t gradUnfoldTransDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t CUDNNWINAPI
cudnnAddTensor(cudnnHandle_t handle,
               const void *alpha,
               const cudnnTensorDescriptor_t aDesc,
               const void *A,
               const void *beta,
               const cudnnTensorDescriptor_t cDesc,
               void *C)
{
    ava_argument(handle) ava_handle;
    ava_argument(alpha) {
        ava_type_cast(const double *);
        ava_in; ava_buffer(1);
    }
    ava_argument(aDesc) ava_handle;
    ava_argument(A) ava_opaque;
    ava_argument(beta) {
        ava_type_cast(const double *);
        ava_in; ava_buffer(1);
    }
    ava_argument(cDesc) ava_handle;
    ava_argument(C) ava_opaque;
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                           cudnnOpTensorOp_t opTensorOp,
                           cudnnDataType_t opTensorCompType,
                           cudnnNanPropagation_t opTensorNanOpt)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                           cudnnOpTensorOp_t *opTensorOp,
                           cudnnDataType_t *opTensorCompType,
                           cudnnNanPropagation_t *opTensorNanOpt)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/* B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
cudnnStatus_t CUDNNWINAPI
cudnnOpTensor(cudnnHandle_t handle,
              const cudnnOpTensorDescriptor_t opTensorDesc,
              const void *alpha1,
              const cudnnTensorDescriptor_t aDesc,
              const void *A,
              const void *alpha2,
              const cudnnTensorDescriptor_t bDesc,
              const void *B,
              const void *beta,
              const cudnnTensorDescriptor_t cDesc,
              void *C)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t reduceTensorOp,
                               cudnnDataType_t reduceTensorCompType,
                               cudnnNanPropagation_t reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t reduceTensorIndices,
                               cudnnIndicesType_t reduceTensorIndicesType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t *reduceTensorOp,
                               cudnnDataType_t *reduceTensorCompType,
                               cudnnNanPropagation_t *reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t *reduceTensorIndices,
                               cudnnIndicesType_t *reduceTensorIndicesType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Helper function to return the minimum size of the index space to be passed to the reduction given the input and
 * output tensors */
cudnnStatus_t CUDNNWINAPI
cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                             const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                             const cudnnTensorDescriptor_t aDesc,
                             const cudnnTensorDescriptor_t cDesc,
                             size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output
 * tensors */
cudnnStatus_t CUDNNWINAPI
cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                               const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               const cudnnTensorDescriptor_t aDesc,
                               const cudnnTensorDescriptor_t cDesc,
                               size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
cudnnStatus_t CUDNNWINAPI
cudnnReduceTensor(cudnnHandle_t handle,
                  const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                  void *indices,
                  size_t indicesSizeInBytes,
                  void *workspace,
                  size_t workspaceSizeInBytes,
                  const void *alpha,
                  const cudnnTensorDescriptor_t aDesc,
                  const void *A,
                  const void *beta,
                  const cudnnTensorDescriptor_t cDesc,
                  void *C)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t CUDNNWINAPI
cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t CUDNNWINAPI
cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format,
                           int k,  /* number of output feature maps */
                           int c,  /* number of input feature maps */
                           int h,  /* height of each input filter */
                           int w)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI
cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format,
                           int *k,  /* number of output feature maps */
                           int *c,  /* number of input feature maps */
                           int *h,  /* height of each input filter */
                           int *w)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI
cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           int nbDimsRequested,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format,
                           int *nbDims,
                           int filterDimA[])
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
cudnnStatus_t CUDNNWINAPI
cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnTransformFilter(cudnnHandle_t handle,
                     const cudnnTensorTransformDescriptor_t transDesc,
                     const void *alpha,
                     const cudnnFilterDescriptor_t srcDesc,
                     const void *srcData,
                     const void *beta,
                     const cudnnFilterDescriptor_t destDesc,
                     void *destData)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnReorderFilterAndBias(cudnnHandle_t handle,
                          const cudnnFilterDescriptor_t filterDesc,
                          cudnnReorderType_t reorderType,
                          const void *filterData,
                          void *reorderedFilterData,
                          int reorderBias,
                          const void *biasData,
                          void *reorderedBiasData)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                int pad_h,      /* zero-padding height */
                                int pad_w,      /* zero-padding width */
                                int u,          /* vertical filter stride */
                                int v,          /* horizontal filter stride */
                                int dilation_h, /* filter dilation in the vertical dimension */
                                int dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t mode,
                                cudnnDataType_t computeType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                int *pad_h,      /* zero-padding height */
                                int *pad_w,      /* zero-padding width */
                                int *u,          /* vertical filter stride */
                                int *v,          /* horizontal filter stride */
                                int *dilation_h, /* filter dilation in the vertical dimension */
                                int *dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t *mode,
                                cudnnDataType_t *computeType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      const cudnnFilterDescriptor_t filterDesc,
                                      int *n,
                                      int *c,
                                      int *h,
                                      int *w)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                int arrayLengthRequested,
                                int *arrayLength,
                                int padA[],
                                int strideA[],
                                int dilationA[],
                                cudnnConvolutionMode_t *mode,
                                cudnnDataType_t *computeType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* convolution data type */

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      const cudnnFilterDescriptor_t filterDesc,
                                      int nbDims,
                                      int tensorOuputDimA[])
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                     const cudnnTensorDescriptor_t xDesc,
                                     const cudnnFilterDescriptor_t wDesc,
                                     const cudnnConvolutionDescriptor_t convDesc,
                                     const cudnnTensorDescriptor_t yDesc,
                                     const int requestedAlgoCount,
                                     int *returnedAlgoCount,
                                     cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#define cu_in_out_buffer(x, y) ({ if(ava_is_in) ava_buffer(x); else ava_buffer(min(x, y == NULL ? x : *y)); })

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnTensorDescriptor_t xDesc,
                                       const void *x,
                                       const cudnnFilterDescriptor_t wDesc,
                                       const void *w,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       const cudnnTensorDescriptor_t yDesc,
                                       void *y,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnConvolutionFwdAlgoPerf_t *perfResults,
                                       void *workSpace,
                                       size_t workSpaceSizeInBytes)
{
    ava_argument(handle) ava_handle;
    ava_argument(xDesc) ava_handle;
    ava_argument(x) ava_opaque;
    ava_argument(wDesc) ava_handle;
    ava_argument(w) ava_opaque;
    ava_argument(convDesc) ava_handle;
    ava_argument(yDesc) ava_handle;
    ava_argument(y) ava_opaque;
    ava_argument(returnedAlgoCount) {
        ava_out; ava_buffer(1);
    }
    ava_argument(perfResults) {
        ava_out; cu_in_out_buffer(requestedAlgoCount, returnedAlgoCount);
    }
    ava_argument(workSpace) ava_opaque;
}

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,
                                      const void *alpha1,
                                      const cudnnTensorDescriptor_t xDesc,
                                      const void *x,
                                      const cudnnFilterDescriptor_t wDesc,
                                      const void *w,
                                      const cudnnConvolutionDescriptor_t convDesc,
                                      cudnnConvolutionFwdAlgo_t algo,
                                      void *workSpace,
                                      size_t workSpaceSizeInBytes,
                                      const void *alpha2,
                                      const cudnnTensorDescriptor_t zDesc,
                                      const void *z,
                                      const cudnnTensorDescriptor_t biasDesc,
                                      const void *bias,
                                      const cudnnActivationDescriptor_t activationDesc,
                                      const cudnnTensorDescriptor_t yDesc,
                                      void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                             const void *alpha,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const void *beta,
                             const cudnnTensorDescriptor_t dbDesc,
                             void *db)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                            const cudnnTensorDescriptor_t xDesc,
                                            const cudnnTensorDescriptor_t dyDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnFilterDescriptor_t dwDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,
                                              const cudnnTensorDescriptor_t xDesc,
                                              const void *x,
                                              const cudnnTensorDescriptor_t dyDesc,
                                              const void *y,
                                              const cudnnConvolutionDescriptor_t convDesc,
                                              const cudnnFilterDescriptor_t dwDesc,
                                              void *dw,
                                              const int requestedAlgoCount,
                                              int *returnedAlgoCount,
                                              cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                              void *workSpace,
                                              size_t workSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t xDesc,
                                           const cudnnTensorDescriptor_t dyDesc,
                                           const cudnnConvolutionDescriptor_t convDesc,
                                           const cudnnFilterDescriptor_t dwDesc,
                                           cudnnConvolutionBwdFilterPreference_t preference,
                                           size_t memoryLimitInBytes,
                                           cudnnConvolutionBwdFilterAlgo_t *algo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                              const cudnnTensorDescriptor_t srcDesc,
                                              const cudnnTensorDescriptor_t diffDesc,
                                              const cudnnConvolutionDescriptor_t convDesc,
                                              const cudnnFilterDescriptor_t gradDesc,
                                              const int requestedAlgoCount,
                                              int *returnedAlgoCount,
                                              cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                               const cudnnTensorDescriptor_t xDesc,
                                               const cudnnTensorDescriptor_t dyDesc,
                                               const cudnnConvolutionDescriptor_t convDesc,
                                               const cudnnFilterDescriptor_t gradDesc,
                                               cudnnConvolutionBwdFilterAlgo_t algo,
                                               size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                               const void *alpha,
                               const cudnnTensorDescriptor_t xDesc,
                               const void *x,
                               const cudnnTensorDescriptor_t dyDesc,
                               const void *dy,
                               const cudnnConvolutionDescriptor_t convDesc,
                               cudnnConvolutionBwdFilterAlgo_t algo,
                               void *workSpace,
                               size_t workSpaceSizeInBytes,
                               const void *beta,
                               const cudnnFilterDescriptor_t dwDesc,
                               void *dw)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                          const cudnnFilterDescriptor_t wDesc,
                                          const cudnnTensorDescriptor_t dyDesc,
                                          const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t dxDesc,
                                          const int requestedAlgoCount,
                                          int *returnedAlgoCount,
                                          cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                            const cudnnFilterDescriptor_t wDesc,
                                            const void *w,
                                            const cudnnTensorDescriptor_t dyDesc,
                                            const void *dy,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t dxDesc,
                                            void *dx,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
                                            void *workSpace,
                                            size_t workSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                         const cudnnFilterDescriptor_t wDesc,
                                         const cudnnTensorDescriptor_t dyDesc,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         const cudnnTensorDescriptor_t dxDesc,
                                         cudnnConvolutionBwdDataPreference_t preference,
                                         size_t memoryLimitInBytes,
                                         cudnnConvolutionBwdDataAlgo_t *algo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
                                            const cudnnFilterDescriptor_t filterDesc,
                                            const cudnnTensorDescriptor_t diffDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t gradDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
                                             const cudnnFilterDescriptor_t wDesc,
                                             const cudnnTensorDescriptor_t dyDesc,
                                             const cudnnConvolutionDescriptor_t convDesc,
                                             const cudnnTensorDescriptor_t dxDesc,
                                             cudnnConvolutionBwdDataAlgo_t algo,
                                             size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBackwardData(cudnnHandle_t handle,
                             const void *alpha,
                             const cudnnFilterDescriptor_t wDesc,
                             const void *w,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const cudnnConvolutionDescriptor_t convDesc,
                             cudnnConvolutionBwdDataAlgo_t algo,
                             void *workSpace,
                             size_t workSpaceSizeInBytes,
                             const void *beta,
                             const cudnnTensorDescriptor_t dxDesc,
                             void *dx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnIm2Col(cudnnHandle_t handle,
            const cudnnTensorDescriptor_t xDesc,
            const void *x,
            const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc,
            void *colBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI
cudnnSoftmaxForward(cudnnHandle_t handle,
                    cudnnSoftmaxAlgorithm_t algo,
                    cudnnSoftmaxMode_t mode,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI
cudnnSoftmaxBackward(cudnnHandle_t handle,
                     cudnnSoftmaxAlgorithm_t algo,
                     cudnnSoftmaxMode_t mode,
                     const void *alpha,
                     const cudnnTensorDescriptor_t yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t dyDesc,
                     const void *dy,
                     const void *beta,
                     const cudnnTensorDescriptor_t dxDesc,
                     void *dx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            cudnnPoolingMode_t mode,
                            cudnnNanPropagation_t maxpoolingNanOpt,
                            int windowHeight,
                            int windowWidth,
                            int verticalPadding,
                            int horizontalPadding,
                            int verticalStride,
                            int horizontalStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                            cudnnPoolingMode_t *mode,
                            cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *windowHeight,
                            int *windowWidth,
                            int *verticalPadding,
                            int *horizontalPadding,
                            int *verticalStride,
                            int *horizontalStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                            int nbDimsRequested,
                            cudnnPoolingMode_t *mode,
                            cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *nbDims,
                            int windowDimA[],
                            int paddingA[],
                            int strideA[])
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims,
                                  int outputTensorDimA[])
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n,
                                  int *c,
                                  int *h,
                                  int *w)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t CUDNNWINAPI
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t mode,
                             cudnnNanPropagation_t reluNanOpt,
                             double coef)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt,
                             double *coef)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI
cudnnActivationForward(cudnnHandle_t handle,
                       cudnnActivationDescriptor_t activationDesc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t xDesc,
                       const void *x,
                       const void *beta,
                       const cudnnTensorDescriptor_t yDesc,
                       void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI
cudnnActivationBackward(cudnnHandle_t handle,
                        cudnnActivationDescriptor_t activationDesc,
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
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/*
* Create an instance of LRN (Local Response Normalization) descriptor
* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
*/
cudnnStatus_t CUDNNWINAPI
cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/*
* Uses a window [center-lookBehind, center+lookAhead], where
* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
* Values of double parameters cast to tensor data type.
*/
cudnnStatus_t CUDNNWINAPI
cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
/*
* Retrieve the settings currently stored in an LRN layer descriptor
* Any of the provided pointers can be NULL (no corresponding value will be returned)
*/
cudnnStatus_t CUDNNWINAPI
cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha, double *lrnBeta, double *lrnK)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Destroy an instance of LRN descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI
cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                            cudnnLRNDescriptor_t normDesc,
                            cudnnLRNMode_t lrnMode,
                            const void *alpha,
                            const cudnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const cudnnTensorDescriptor_t yDesc,
                            void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI
cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
                             cudnnLRNDescriptor_t normDesc,
                             cudnnLRNMode_t lrnMode,
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
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t CUDNNWINAPI
cudnnDivisiveNormalizationForward(cudnnHandle_t handle,
                                  cudnnLRNDescriptor_t normDesc,
                                  cudnnDivNormMode_t mode,
                                  const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
                                  const void *x,
                                  const void *means, /* if NULL, means are assumed to be zero */
                                  void *temp,
                                  void *temp2,
                                  const void *beta,
                                  const cudnnTensorDescriptor_t yDesc,
                                  void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDivisiveNormalizationBackward(cudnnHandle_t handle,
                                   cudnnLRNDescriptor_t normDesc,
                                   cudnnDivNormMode_t mode,
                                   const void *alpha,
                                   const cudnnTensorDescriptor_t xDesc, /* same desc for x, means, dy, temp, temp2 */
                                   const void *x,
                                   const void *means, /* if NULL, means are assumed to be zero */
                                   const void *dy,
                                   void *temp,
                                   void *temp2,
                                   const void *beta,
                                   const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
                                   void *dx,                                   /* output x differential */
                                   void *dMeans)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* output means differential, can be NULL */


/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
*/
cudnnStatus_t CUDNNWINAPI
cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t xDesc,
                              cudnnBatchNormMode_t mode)
{
    ava_argument(derivedBnDesc) ava_handle;
    ava_argument(xDesc) ava_handle;
}

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle,
                                                  cudnnBatchNormMode_t mode,
                                                  cudnnBatchNormOps_t bnOps,
                                                  const cudnnTensorDescriptor_t xDesc,
                                                  const cudnnTensorDescriptor_t yDesc,
                                                  const cudnnTensorDescriptor_t dyDesc,
                                                  const cudnnTensorDescriptor_t dzDesc,
                                                  const cudnnTensorDescriptor_t dxDesc,
                                                  const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                                  const cudnnActivationDescriptor_t activationDesc,
                                                  size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle,
                                                     cudnnBatchNormMode_t mode,
                                                     cudnnBatchNormOps_t bnOps,
                                                     const cudnnActivationDescriptor_t activationDesc,
                                                     const cudnnTensorDescriptor_t xDesc,
                                                     size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc,
    const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc,
    void *y, /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
        (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
    const void *bnScale,
    const void *bnBias,

    /* MUST use factor=1 in the very first call of a complete training cycle.
       Use a factor=1/(1+n) at N-th call to the function to get
       Cumulative Moving Average (CMA) behavior
       CMA[n] = (x[1]+...+x[n])/n
       Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
       ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
       CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
    double exponentialAverageFactor,

    /* Used in Training phase only.
       runningMean = newMean*factor + runningMean*(1-factor) */
    void *resultRunningMean,
    /* Output in training mode, input in inference. Is the moving average
       of  variance[x] (factor is applied in the same way as for runningMean) */
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Computes y = relu(BN(x) + z). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,

    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,

    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance,

    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationBackward(cudnnHandle_t handle,
                                cudnnBatchNormMode_t mode,
                                const void *alphaDataDiff,
                                const void *betaDataDiff,
                                const void *alphaParamDiff,
                                const void *betaParamDiff,
                                const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
                                const void *x,
                                const cudnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const cudnnTensorDescriptor_t dxDesc,
                                void *dx,
                                /* Shared tensor desc for the 4 tensors below */
                                const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                const void *bnScale, /* bnBias doesn't affect backpropagation */
                                /* scale and bias diff are not backpropagated below this layer */
                                void *dBnScaleResult,
                                void *dBnBiasResult,
                                /* Same epsilon as forward pass */
                                double epsilon,

                                /* Optionally cached intermediate results from
                                   forward pass */
                                const void *savedMean,
                                const void *savedInvVariance)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,
                                  cudnnBatchNormMode_t mode,
                                  cudnnBatchNormOps_t bnOps,

                                  const void *alphaDataDiff,
                                  const void *betaDataDiff,
                                  const void *alphaParamDiff,
                                  const void *betaParamDiff,
                                  const cudnnTensorDescriptor_t xDesc,
                                  const void *xData,
                                  const cudnnTensorDescriptor_t yDesc,
                                  const void *yData,
                                  const cudnnTensorDescriptor_t dyDesc,
                                  const void *dyData,
                                  const cudnnTensorDescriptor_t dzDesc,
                                  void *dzData,
                                  const cudnnTensorDescriptor_t dxDesc,
                                  void *dxData,

                                  /* Shared tensor desc for the 4 tensors below */
                                  const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                  const void *bnScaleData,
                                  const void *bnBiasData, /* needed if there is activation */
                                  void *dBnScaleData,
                                  void *dBnBiasData,
                                  double epsilon, /* Same epsilon as forward pass */

                                  /* Optionally cached intermediate results from
                                     forward pass */
                                  const void *savedMean,
                                  const void *savedInvVariance,
                                  cudnnActivationDescriptor_t activationDesc,
                                  void *workSpace,
                                  size_t workSpaceSizeInBytes,
                                  void *reserveSpace,
                                  size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                       cudnnSamplerType_t samplerType,
                                       cudnnDataType_t dataType,
                                       const int nbDims,
                                       const int dimA[])
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                   const cudnnSpatialTransformerDescriptor_t stDesc,
                                   const void *theta,
                                   void *grid)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                    const cudnnSpatialTransformerDescriptor_t stDesc,
                                    const void *dgrid,
                                    void *dtheta)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfSamplerForward(cudnnHandle_t handle,
                             cudnnSpatialTransformerDescriptor_t stDesc,
                             const void *alpha,
                             const cudnnTensorDescriptor_t xDesc,
                             const void *x,
                             const void *grid,
                             const void *beta,
                             cudnnTensorDescriptor_t yDesc,
                             void *y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
                              cudnnSpatialTransformerDescriptor_t stDesc,
                              const void *alpha,
                              const cudnnTensorDescriptor_t xDesc,
                              const void *x,
                              const void *beta,
                              const cudnnTensorDescriptor_t dxDesc,
                              void *dx,
                              const void *alphaDgrid,
                              const cudnnTensorDescriptor_t dyDesc,
                              const void *dy,
                              const void *grid,
                              const void *betaDgrid,
                              void *dgrid)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cudnnStatus_t CUDNNWINAPI
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
cudnnStatus_t CUDNNWINAPI
cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                          cudnnHandle_t handle,
                          float dropout,
                          void *states,
                          size_t stateSizeInBytes,
                          unsigned long long seed)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Restores the dropout descriptor to a previously saved-off state */
cudnnStatus_t CUDNNWINAPI
cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                              cudnnHandle_t handle,
                              float dropout,
                              void *states,
                              size_t stateSizeInBytes,
                              unsigned long long seed)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                          cudnnHandle_t handle,
                          float *dropout,
                          void **states,
                          unsigned long long *seed)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDropoutForward(cudnnHandle_t handle,
                    const cudnnDropoutDescriptor_t dropoutDesc,
                    const cudnnTensorDescriptor_t xdesc,
                    const void *x,
                    const cudnnTensorDescriptor_t ydesc,
                    void *y,
                    void *reserveSpace,
                    size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDropoutBackward(cudnnHandle_t handle,
                     const cudnnDropoutDescriptor_t dropoutDesc,
                     const cudnnTensorDescriptor_t dydesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t dxdesc,
                     void *dx,
                     void *reserveSpace,
                     size_t reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Sequence data descriptor */

cudnnStatus_t CUDNNWINAPI
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t dataType,
                          int nbDims,
                          const int dimA[],
                          const cudnnSeqDataAxis_t axes[],
                          size_t seqLengthArraySize,
                          const int seqLengthArray[],
                          void *paddingFill)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t *dataType,
                          int *nbDims,
                          int nbDimsRequested,
                          int dimA[],
                          cudnnSeqDataAxis_t axes[],
                          size_t *seqLengthArraySize,
                          size_t seqLengthSizeRequested,
                          int seqLengthArray[],
                          void *paddingFill)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* Multihead Attention */

/* Multi-head attention modes set in attention descriptor */

cudnnStatus_t CUDNNWINAPI
cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       unsigned attnMode,
                       int nHeads,
                       double smScaler,
                       cudnnDataType_t dataType,
                       cudnnDataType_t computePrec,
                       cudnnMathType_t mathType,
                       cudnnDropoutDescriptor_t attnDropoutDesc,
                       cudnnDropoutDescriptor_t postDropoutDesc,
                       int qSize,
                       int kSize,
                       int vSize,
                       int qProjSize,
                       int kProjSize,
                       int vProjSize,
                       int oProjSize,
                       int qoMaxSeqLength,
                       int kvMaxSeqLength,
                       int maxBatchSize,
                       int maxBeamSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       unsigned *attnMode,
                       int *nHeads,
                       double *smScaler,
                       cudnnDataType_t *dataType,
                       cudnnDataType_t *computePrec,
                       cudnnMathType_t *mathType,
                       cudnnDropoutDescriptor_t *attnDropoutDesc,
                       cudnnDropoutDescriptor_t *postDropoutDesc,
                       int *qSize,
                       int *kSize,
                       int *vSize,
                       int *qProjSize,
                       int *kProjSize,
                       int *vProjSize,
                       int *oProjSize,
                       int *qoMaxSeqLength,
                       int *kvMaxSeqLength,
                       int *maxBatchSize,
                       int *maxBeamSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             size_t *weightSizeInBytes,
                             size_t *workSpaceSizeInBytes,
                             size_t *reserveSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             cudnnMultiHeadAttnWeightKind_t wKind,
                             size_t weightSizeInBytes,
                             const void *weights,
                             cudnnTensorDescriptor_t wDesc,
                             void **wAddr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnMultiHeadAttnForward(cudnnHandle_t handle,
                          const cudnnAttnDescriptor_t attnDesc,
                          int currIdx,
                          const int loWinIdx[],
                          const int hiWinIdx[],
                          const int devSeqLengthsQO[],
                          const int devSeqLengthsKV[],
                          const cudnnSeqDataDescriptor_t qDesc,
                          const void *queries,
                          const void *residuals,
                          const cudnnSeqDataDescriptor_t kDesc,
                          const void *keys,
                          const cudnnSeqDataDescriptor_t vDesc,
                          const void *values,
                          const cudnnSeqDataDescriptor_t oDesc,
                          void *out,
                          size_t weightSizeInBytes,
                          const void *weights,
                          size_t workSpaceSizeInBytes,
                          void *workSpace,
                          size_t reserveSpaceSizeInBytes,
                          void *reserveSpace)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle,
                               const cudnnAttnDescriptor_t attnDesc,
                               const int loWinIdx[],
                               const int hiWinIdx[],
                               const int devSeqLengthsDQDO[],
                               const int devSeqLengthsDKDV[],
                               const cudnnSeqDataDescriptor_t doDesc,
                               const void *dout,
                               const cudnnSeqDataDescriptor_t dqDesc,
                               void *dqueries,
                               const void *queries,
                               const cudnnSeqDataDescriptor_t dkDesc,
                               void *dkeys,
                               const void *keys,
                               const cudnnSeqDataDescriptor_t dvDesc,
                               void *dvalues,
                               const void *values,
                               size_t weightSizeInBytes,
                               const void *weights,
                               size_t workSpaceSizeInBytes,
                               void *workSpace,
                               size_t reserveSpaceSizeInBytes,
                               void *reserveSpace)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle,
                                  const cudnnAttnDescriptor_t attnDesc,
                                  cudnnWgradMode_t addGrad,
                                  const cudnnSeqDataDescriptor_t qDesc,
                                  const void *queries,
                                  const cudnnSeqDataDescriptor_t kDesc,
                                  const void *keys,
                                  const cudnnSeqDataDescriptor_t vDesc,
                                  const void *values,
                                  const cudnnSeqDataDescriptor_t doDesc,
                                  const void *dout,
                                  size_t weightSizeInBytes,
                                  const void *weights,
                                  void *dweights,
                                  size_t workSpaceSizeInBytes,
                                  void *workSpace,
                                  size_t reserveSpaceSizeInBytes,
                                  void *reserveSpace)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/*
* CTC (Connectionist Temporal Classification) loss descriptor create/destory/set/get functions
*/
cudnnStatus_t CUDNNWINAPI
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                            cudnnDataType_t compType,
                            cudnnLossNormalizationMode_t normMode,
                            cudnnNanPropagation_t gradMode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                            cudnnDataType_t *compType,
                            cudnnLossNormalizationMode_t *normMode,
                            cudnnNanPropagation_t *gradMode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t CUDNNWINAPI
cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the
                          mini batch size, A is the alphabet size)  */
    const void *probs, /* probabilities after softmax, in GPU memory */
    const int *labels, /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    void *costs,                                 /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
    const void *gradients,   /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,              /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* size of the workspace */

/* return the workspace size needed for ctc */
cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size) */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the
                                                    dimensions are T,N,A. To compute costs
                                                    only, set it to NULL */
    const int *labels,                           /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    cudnnCTCLossAlgo_t algo,                     /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
} /* pointer to the returned workspace size */

cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t algoDesc,
                             cudnnStatus_t status,
                             float time,
                             size_t memory)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t *algoDesc,
                             cudnnStatus_t *status,
                             float *time,
                             size_t *memory)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSaveAlgorithm(cudnnHandle_t handle,
                   cudnnAlgorithmDescriptor_t algoDesc,
                   void *algoSpace,
                   size_t algoSpaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnRestoreAlgorithm(cudnnHandle_t handle,
                      void *algoSpace,
                      size_t algoSpaceSizeInBytes,
                      cudnnAlgorithmDescriptor_t algoDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,
                                        cudnnFusedOpsConstParamLabel_t paramLabel,
                                        const void *param)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                        cudnnFusedOpsConstParamLabel_t paramLabel,
                                        void *param,
                                        int *isNULL)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          void *ptr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          void *ptr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnMakeFusedOpsPlan(cudnnHandle_t handle,
                      cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack,
                      size_t *workspaceSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cudnnStatus_t CUDNNWINAPI
cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/******** curand *********/
curandStatus_t CURANDAPI
curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type)
{
    ava_argument(generator) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

curandStatus_t CURANDAPI
curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandDestroyGenerator(curandGenerator_t generator)
{
    ava_argument(generator) ava_handle;
}

curandStatus_t CURANDAPI
curandGetVersion(int *version)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGetProperty(libraryPropertyType type, int *value)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


curandStatus_t CURANDAPI
curandSetStream(curandGenerator_t generator, cudaStream_t stream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr, size_t num)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateNormal(curandGenerator_t generator, float *outputPtr,
                     size_t n, float mean, float stddev)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr,
                     size_t n, double mean, double stddev)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr,
                     size_t n, float mean, float stddev)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr,
                     size_t n, double mean, double stddev)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

curandStatus_t CURANDAPI
curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *discrete_distribution)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


curandStatus_t CURANDAPI
curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


curandStatus_t CURANDAPI
curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr,
                     size_t n, double lambda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

// just for internal usage
curandStatus_t CURANDAPI
curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int *outputPtr,
                     size_t n, double lambda, curandMethod_t method)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


curandStatus_t CURANDAPI
curandGenerateBinomial(curandGenerator_t generator, unsigned int *outputPtr,
                       size_t num, unsigned int n, double p)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}
// just for internal usage
curandStatus_t CURANDAPI
curandGenerateBinomialMethod(curandGenerator_t generator,
                             unsigned int *outputPtr,
                             size_t num, unsigned int n, double p,
                             curandMethod_t method)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


curandStatus_t CURANDAPI
curandGenerateSeeds(curandGenerator_t generator)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

// curandStatus_t CURANDAPI
// curandGetDirectionVectors32( unsigned int (*vectors[32])[], curandDirectionVectorSet_t set)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }

curandStatus_t CURANDAPI
curandGetScrambleConstants32(unsigned int ** constants)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

// curandStatus_t CURANDAPI
// curandGetDirectionVectors64(unsigned long long (*vectors[64])[], curandDirectionVectorSet_t set)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }

curandStatus_t CURANDAPI
curandGetScrambleConstants64(unsigned long long * * constants)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/******** cufft *********/
cufftResult CUFFTAPI cufftPlan1d(cufftHandle *plan,
                                 int nx,
                                 cufftType type,
                                 int batch)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftPlan2d(cufftHandle *plan,
                                 int nx, int ny,
                                 cufftType type)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftPlan3d(cufftHandle *plan,
                                 int nx, int ny, int nz,
                                 cufftType type)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftPlanMany(cufftHandle *plan,
                                   int rank,
                                   int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   cufftType type,
                                   int batch)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftMakePlan1d(cufftHandle plan,
                                     int nx,
                                     cufftType type,
                                     int batch,
                                     size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftMakePlan2d(cufftHandle plan,
                                     int nx, int ny,
                                     cufftType type,
                                     size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftMakePlan3d(cufftHandle plan,
                                     int nx, int ny, int nz,
                                     cufftType type,
                                     size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftMakePlanMany(cufftHandle plan,
                                       int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       cufftType type,
                                       int batch,
                                       size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftMakePlanMany64(cufftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         long long int *onembed,
                                         long long int ostride, long long int odist,
                                         cufftType type,
                                         long long int batch,
                                         size_t * workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetSizeMany64(cufftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride, long long int idist,
                                        long long int *onembed,
                                        long long int ostride, long long int odist,
                                        cufftType type,
                                        long long int batch,
                                        size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftEstimate1d(int nx,
                                     cufftType type,
                                     int batch,
                                     size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftEstimate2d(int nx, int ny,
                                     cufftType type,
                                     size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftEstimate3d(int nx, int ny, int nz,
                                     cufftType type,
                                     size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftEstimateMany(int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       cufftType type,
                                       int batch,
                                       size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftCreate(cufftHandle * handle)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetSize1d(cufftHandle handle,
                                    int nx,
                                    cufftType type,
                                    int batch,
                                    size_t *workSize )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetSize2d(cufftHandle handle,
                                    int nx, int ny,
                                    cufftType type,
                                    size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetSize3d(cufftHandle handle,
                                    int nx, int ny, int nz,
                                    cufftType type,
                                    size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetSizeMany(cufftHandle handle,
                                      int rank, int *n,
                                      int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist,
                                      cufftType type, int batch, size_t *workArea)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetSize(cufftHandle handle, size_t *workSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftSetWorkArea(cufftHandle plan, void *workArea)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftSetAutoAllocation(cufftHandle plan, int autoAllocate)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftExecC2C(cufftHandle plan,
                                  cufftComplex *idata,
                                  cufftComplex *odata,
                                  int direction)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftExecR2C(cufftHandle plan,
                                  cufftReal *idata,
                                  cufftComplex *odata)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftExecC2R(cufftHandle plan,
                                  cufftComplex *idata,
                                  cufftReal *odata)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftExecZ2Z(cufftHandle plan,
                                  cufftDoubleComplex *idata,
                                  cufftDoubleComplex *odata,
                                  int direction)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftExecD2Z(cufftHandle plan,
                                  cufftDoubleReal *idata,
                                  cufftDoubleComplex *odata)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftExecZ2D(cufftHandle plan,
                                  cufftDoubleComplex *idata,
                                  cufftDoubleReal *odata)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


// utility functions
cufftResult CUFFTAPI cufftSetStream(cufftHandle plan,
                                    cudaStream_t stream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftDestroy(cufftHandle plan)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetVersion(int *version)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cufftResult CUFFTAPI cufftGetProperty(libraryPropertyType type,
                                      int *value)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/******* cusolver *********/
cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cusolverStatus_t CUSOLVERAPI cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* Cholesky factorization and its solver */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *Lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}




cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const float *A,
    int lda,
    float *B,
    int ldb,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const double *A,
    int lda,
    double *B,
    int ldb,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* batched Cholesky factorization and its solver */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *Aarray[],
    int lda,
    int *infoArray,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *Aarray[],
    int lda,
    int *infoArray,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    float *A[],
    int lda,
    float *B[],
    int ldb,
    int *d_info,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    double *A[],
    int lda,
    double *B[],
    int ldb,
    int *d_info,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    cuComplex *A[],
    int lda,
    cuComplex *B[],
    int ldb,
    int *d_info,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    cuDoubleComplex *A[],
    int lda,
    cuDoubleComplex *B[],
    int ldb,
    int *d_info,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* s.p.d. matrix inversion (POTRI) and auxiliary routines (TRTRI and LAUUM)  */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnStrtri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    float *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDtrtri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    double *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCtrtri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    cuComplex *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZtrtri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnStrtri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDtrtri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCtrtri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZtrtri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* lauum, auxiliar routine for s.p.d matrix inversion */
cusolverStatus_t CUSOLVERAPI cusolverDnSlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnClauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnClauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}




/* LU Factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    float *Workspace,
    int *devIpiv,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    double *Workspace,
    int *devIpiv,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *Workspace,
    int *devIpiv,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *Workspace,
    int *devIpiv,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* Row pivoting */
cusolverStatus_t CUSOLVERAPI cusolverDnSlaswp(
    cusolverDnHandle_t handle,
    int n,
    float *A,
    int lda,
    int k1,
    int k2,
    const int *devIpiv,
    int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDlaswp(
    cusolverDnHandle_t handle,
    int n,
    double *A,
    int lda,
    int k1,
    int k2,
    const int *devIpiv,
    int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnClaswp(
    cusolverDnHandle_t handle,
    int n,
    cuComplex *A,
    int lda,
    int k1,
    int k2,
    const int *devIpiv,
    int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZlaswp(
    cusolverDnHandle_t handle,
    int n,
    cuDoubleComplex *A,
    int lda,
    int k1,
    int k2,
    const int *devIpiv,
    int incx)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* LU solve */
cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs(
    cusolverDnHandle_t handle,
    cublasOperation_t trans,
    int n,
    int nrhs,
    const float *A,
    int lda,
    const int *devIpiv,
    float *B,
    int ldb,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs(
    cusolverDnHandle_t handle,
    cublasOperation_t trans,
    int n,
    int nrhs,
    const double *A,
    int lda,
    const int *devIpiv,
    double *B,
    int ldb,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgetrs(
    cusolverDnHandle_t handle,
    cublasOperation_t trans,
    int n,
    int nrhs,
    const cuComplex *A,
    int lda,
    const int *devIpiv,
    cuComplex *B,
    int ldb,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgetrs(
    cusolverDnHandle_t handle,
    cublasOperation_t trans,
    int n,
    int nrhs,
    const cuDoubleComplex *A,
    int lda,
    const int *devIpiv,
    cuDoubleComplex *B,
    int ldb,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* QR factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    float *TAU,
    float *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    double *TAU,
    double *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *TAU,
    cuComplex *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *TAU,
    cuDoubleComplex *Workspace,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* generate unitary matrix Q from QR factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCungqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZungqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCungqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZungqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}




/* compute Q**T*b in solve min||A*x = b|| */
cusolverStatus_t CUSOLVERAPI cusolverDnSormqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDormqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    const cuComplex *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    const cuDoubleComplex *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* L*D*L**T,U*D*U**T factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    float *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    double *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    cuComplex *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *ipiv,
    float *work,
    int lwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *ipiv,
    double *work,
    int lwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *ipiv,
    cuComplex *work,
    int lwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *ipiv,
    cuDoubleComplex *work,
    int lwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* Symmetric indefinite solve (SYTRS) */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrs_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const float *A,
        int lda,
        const int *ipiv,
        float *B,
        int ldb,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytrs_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const double *A,
        int lda,
        const int *ipiv,
        double *B,
        int ldb,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCsytrs_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuComplex *A,
        int lda,
        const int *ipiv,
        cuComplex *B,
        int ldb,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZsytrs_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        cuDoubleComplex *B,
        int ldb,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSsytrs(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const float *A,
        int lda,
        const int *ipiv,
        float *B,
        int ldb,
        float *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytrs(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const double *A,
        int lda,
        const int *ipiv,
        double *B,
        int ldb,
        double *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCsytrs(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuComplex *A,
        int lda,
        const int *ipiv,
        cuComplex *B,
        int ldb,
        cuComplex *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZsytrs(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        cuDoubleComplex *B,
        int ldb,
        cuDoubleComplex *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* Symmetric indefinite inversion (sytri) */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

;
cusolverStatus_t CUSOLVERAPI cusolverDnCsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex *A,
        int lda,
        const int *ipiv,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        float *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        double *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex *A,
        int lda,
        const int *ipiv,
        cuComplex *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        cuDoubleComplex *work,
        int lwork,
        int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* bidiagonal factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    float *D,
    float *E,
    float *TAUQ,
    float *TAUP,
    float *Work,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    double *D,
    double *E,
    double *TAUQ,
    double *TAUP,
    double *Work,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    float *D,
    float *E,
    cuComplex *TAUQ,
    cuComplex *TAUP,
    cuComplex *Work,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *D,
    double *E,
    cuDoubleComplex *TAUQ,
    cuDoubleComplex *TAUP,
    cuDoubleComplex *Work,
    int Lwork,
    int *devInfo )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* generates one of the unitary matrices Q or P**T determined by GEBRD*/
cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCungbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZungbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCungbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZungbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    int m,
    int n,
    int k,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* tridiagonal factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *d,
    const float *e,
    const float *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *d,
    const double *e,
    const double *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChetrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *d,
    const float *e,
    const cuComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *d,
    const double *e,
    const cuDoubleComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *d,
    float *e,
    float *tau,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *d,
    double *e,
    double *tau,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChetrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *d,
    float *e,
    cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *d,
    double *e,
    cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}




/* generate unitary Q comes from sytrd */
cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCungtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZungtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCungtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZungtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}




/* compute op(Q)*C or C*op(Q) where Q comes from sytrd */
cusolverStatus_t CUSOLVERAPI cusolverDnSormtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDormtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    const cuComplex *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    const cuDoubleComplex *C,
    int ldc,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSormtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    float *A,
    int lda,
    float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDormtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    double *A,
    int lda,
    double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}




/* singular value decomposition, A = U * Sigma * V^H */
cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd (
    cusolverDnHandle_t handle,
    signed char jobu,
    signed char jobvt,
    int m,
    int n,
    float *A,
    int lda,
    float *S,
    float *U,
    int ldu,
    float *VT,
    int ldvt,
    float *work,
    int lwork,
    float *rwork,
    int  *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd (
    cusolverDnHandle_t handle,
    signed char jobu,
    signed char jobvt,
    int m,
    int n,
    double *A,
    int lda,
    double *S,
    double *U,
    int ldu,
    double *VT,
    int ldvt,
    double *work,
    int lwork,
    double *rwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd (
    cusolverDnHandle_t handle,
    signed char jobu,
    signed char jobvt,
    int m,
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *VT,
    int ldvt,
    cuComplex *work,
    int lwork,
    float *rwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd (
    cusolverDnHandle_t handle,
    signed char jobu,
    signed char jobvt,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *S,
    cuDoubleComplex *U,
    int ldu,
    cuDoubleComplex *VT,
    int ldvt,
    cuDoubleComplex *work,
    int lwork,
    double *rwork,
    int *info )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* standard symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *W,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* standard selective symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* selective generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *B,
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *B,
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B,
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *B,
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *B,
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *B,
    int ldb,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChegvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *B,
    int ldb,
    const float *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B,
    int ldb,
    const double *W,
    int *lwork)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *B,
    int ldb,
    float *W,
    float *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *B,
    int ldb,
    double *W,
    double *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChegvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    float *W,
    cuComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnCreateSyevjInfo(
    syevjInfo_t *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDestroySyevjInfo(
    syevjInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetTolerance(
    syevjInfo_t info,
    double tolerance)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetMaxSweeps(
    syevjInfo_t info,
    int max_sweeps)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetSortEig(
    syevjInfo_t info,
    int sort_eig)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetResidual(
    cusolverDnHandle_t handle,
    syevjInfo_t info,
    double *residual)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetSweeps(
    cusolverDnHandle_t handle,
    syevjInfo_t info,
    int *executed_sweeps)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnCheevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnZheevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnCheevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnZheevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *B,
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChegvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *B,
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B,
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *B,
    int ldb,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *B,
    int ldb,
    double *W,
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnChegvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnCreateGesvdjInfo(
    gesvdjInfo_t *info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDestroyGesvdjInfo(
    gesvdjInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetTolerance(
    gesvdjInfo_t info,
    double tolerance)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetMaxSweeps(
    gesvdjInfo_t info,
    int max_sweeps)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetSortEig(
    gesvdjInfo_t info,
    int sort_svd)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetResidual(
    cusolverDnHandle_t handle,
    gesvdjInfo_t info,
    double *residual)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetSweeps(
    cusolverDnHandle_t handle,
    gesvdjInfo_t info,
    int *executed_sweeps)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const float *A,
    int lda,
    const float *S,
    const float *U,
    int ldu,
    const float *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const double *A,
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const float *S,
    const cuComplex *U,
    int ldu,
    const cuComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *S,
    const cuDoubleComplex *U,
    int ldu,
    const cuDoubleComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    float *A,
    int lda,
    float *S,
    float *U,
    int ldu,
    float *V,
    int ldv,
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    double *A,
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv,
    double *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *V,
    int ldv,
    cuComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *S,
    cuDoubleComplex *U,
    int ldu,
    cuDoubleComplex *V,
    int ldv,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const float *A,
    int lda,
    const float *S,
    const float *U,
    int ldu,
    const float *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const double *A,
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const float *S,
    const cuComplex *U,
    int ldu,
    const cuComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *S,
    const cuDoubleComplex *U,
    int ldu,
    const cuDoubleComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    float *A,
    int lda,
    float *S,
    float *U,
    int ldu,
    float *V,
    int ldv,
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    double *A,
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv,
    double *work,
    int lwork,
    int *info,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *V,
    int ldv,
    cuComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *S,
    cuDoubleComplex *U,
    int ldu,
    cuDoubleComplex *V,
    int ldv,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



/* batched approximate SVD */

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const float *d_A,
    int lda,
    long long int strideA,
    const float *d_S,
    long long int strideS,
    const float *d_U,
    int ldu,
    long long int strideU,
    const float *d_V,
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const double *d_A,
    int lda,
    long long int strideA,
    const double *d_S,
    long long int strideS,
    const double *d_U,
    int ldu,
    long long int strideU,
    const double *d_V,
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuComplex *d_A,
    int lda,
    long long int strideA,
    const float *d_S,
    long long int strideS,
    const cuComplex *d_U,
    int ldu,
    long long int strideU,
    const cuComplex *d_V,
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuDoubleComplex *d_A,
    int lda,
    long long int strideA,
    const double *d_S,
    long long int strideS,
    const cuDoubleComplex *d_U,
    int ldu,
    long long int strideU,
    const cuDoubleComplex *d_V,
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const float *d_A,
    int lda,
    long long int strideA,
    float *d_S,
    long long int strideS,
    float *d_U,
    int ldu,
    long long int strideU,
    float *d_V,
    int ldv,
    long long int strideV,
    float *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const double *d_A,
    int lda,
    long long int strideA,
    double *d_S,
    long long int strideS,
    double *d_U,
    int ldu,
    long long int strideU,
    double *d_V,
    int ldv,
    long long int strideV,
    double *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuComplex *d_A,
    int lda,
    long long int strideA,
    float *d_S,
    long long int strideS,
    cuComplex *d_U,
    int ldu,
    long long int strideU,
    cuComplex *d_V,
    int ldv,
    long long int strideV,
    cuComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}



cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuDoubleComplex *d_A,
    int lda,
    long long int strideA,
    double *d_S,
    long long int strideS,
    cuDoubleComplex *d_U,
    int ldu,
    long long int strideU,
    cuDoubleComplex *d_V,
    int ldv,
    long long int strideV,
    cuDoubleComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/******* cusparse *********/
//##############################################################################
//# INITILIAZATION AND MANAGMENT ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreate(cusparseHandle_t* handle)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroy(cusparseHandle_t handle)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseGetVersion(cusparseHandle_t handle,
                   int*             version)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseGetProperty(libraryPropertyType type,
                    int*                value)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetStream(cusparseHandle_t handle,
                  cudaStream_t     streamId)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseGetStream(cusparseHandle_t handle,
                  cudaStream_t*    streamId)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseGetPointerMode(cusparseHandle_t       handle,
                       cusparsePointerMode_t* mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetPointerMode(cusparseHandle_t      handle,
                       cusparsePointerMode_t mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# HELPER ROUTINES
//##############################################################################

const char* CUSPARSEAPI
cusparseGetErrorName(cusparseStatus_t status)
{
    const char *ret = ava_execute();
    ava_return_value {
        ava_out; ava_buffer(strlen(ret) + 1);
        ava_lifetime_static;
    }
}


const char* CUSPARSEAPI
cusparseGetErrorString(cusparseStatus_t status)
{
    const char *ret = ava_execute();
    ava_return_value {
        ava_out; ava_buffer(strlen(ret) + 1);
        ava_lifetime_static;
    }
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateMatDescr(cusparseMatDescr_t* descrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyMatDescr(cusparseMatDescr_t descrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCopyMatDescr(cusparseMatDescr_t       dest,
                     const cusparseMatDescr_t src)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetMatType(cusparseMatDescr_t   descrA,
                   cusparseMatrixType_t type)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseMatrixType_t CUSPARSEAPI
cusparseGetMatType(const cusparseMatDescr_t descrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                       cusparseFillMode_t fillMode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseFillMode_t CUSPARSEAPI
cusparseGetMatFillMode(const cusparseMatDescr_t descrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                       cusparseDiagType_t diagType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseDiagType_t CUSPARSEAPI
cusparseGetMatDiagType(const cusparseMatDescr_t descrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetMatIndexBase(cusparseMatDescr_t  descrA,
                        cusparseIndexBase_t base_)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseIndexBase_t CUSPARSEAPI
cusparseGetMatIndexBase(const cusparseMatDescr_t descrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseGetLevelInfo(cusparseHandle_t            handle,
                     cusparseSolveAnalysisInfo_t info,
                     int*                        nlevels,
                     int**                       levelPtr,
                     int**                       levelInd)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCsric02Info(csric02Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsric02Info(csric02Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateBsric02Info(bsric02Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsric02Info(bsric02Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrilu02Info(csrilu02Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrilu02Info(bsrilu02Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrsv2Info(bsrsv2Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrsm2Info(bsrsm2Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateHybMat(cusparseHybMat_t* hybA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyHybMat(cusparseHybMat_t hybA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCsru2csrInfo(csru2csrInfo_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateColorInfo(cusparseColorInfo_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyColorInfo(cusparseColorInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSetColorAlgs(cusparseColorInfo_t info,
                     cusparseColorAlg_t  alg)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseGetColorAlgs(cusparseColorInfo_t info,
                     cusparseColorAlg_t* alg)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreatePruneInfo(pruneInfo_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyPruneInfo(pruneInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE LEVEL 1 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSaxpyi(cusparseHandle_t    handle,
               int                 nnz,
               const float*        alpha,
               const float*        xVal,
               const int*          xInd,
               float*              y,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDaxpyi(cusparseHandle_t    handle,
               int                 nnz,
               const double*       alpha,
               const double*       xVal,
               const int*          xInd,
               double*             y,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCaxpyi(cusparseHandle_t    handle,
               int                 nnz,
               const cuComplex*    alpha,
               const cuComplex*    xVal,
               const int*          xInd,
               cuComplex*          y,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZaxpyi(cusparseHandle_t       handle,
               int                    nnz,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               cuDoubleComplex*       y,
               cusparseIndexBase_t    idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgthr(cusparseHandle_t    handle,
              int                 nnz,
              const float*        y,
              float*              xVal,
              const int*          xInd,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgthr(cusparseHandle_t    handle,
              int                 nnz,
              const double*       y,
              double*             xVal,
              const int*          xInd,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgthr(cusparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    y,
              cuComplex*          xVal,
              const int*          xInd,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgthr(cusparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* y,
              cuDoubleComplex*       xVal,
              const int*             xInd,
              cusparseIndexBase_t    idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgthrz(cusparseHandle_t    handle,
               int                 nnz,
               float*              y,
               float*              xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgthrz(cusparseHandle_t    handle,
               int                 nnz,
               double*             y,
               double*             xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgthrz(cusparseHandle_t    handle,
               int                 nnz,
               cuComplex*          y,
               cuComplex*          xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgthrz(cusparseHandle_t    handle,
               int                 nnz,
               cuDoubleComplex*    y,
               cuDoubleComplex*    xVal,
               const int*          xInd,
               cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSsctr(cusparseHandle_t    handle,
              int                 nnz,
              const float*        xVal,
              const int*          xInd,
              float*              y,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDsctr(cusparseHandle_t    handle,
              int                 nnz,
              const double*       xVal,
              const int*          xInd,
              double*             y,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCsctr(cusparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    xVal,
              const int*          xInd,
              cuComplex*          y,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZsctr(cusparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* xVal,
              const int*             xInd,
              cuDoubleComplex*       y,
              cusparseIndexBase_t    idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSroti(cusparseHandle_t    handle,
              int                 nnz,
              float*              xVal,
              const int*          xInd,
              float*              y,
              const float*        c,
              const float*        s,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDroti(cusparseHandle_t    handle,
              int                 nnz,
              double*             xVal,
              const int*          xInd,
              double*             y,
              const double*       c,
              const double*       s,
              cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE LEVEL 2 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi(cusparseHandle_t    handle,
               cusparseOperation_t transA,
               int                 m,
               int                 n,
               const float*        alpha,
               const float*        A,
               int                 lda,
               int                 nnz,
               const float*        xVal,
               const int*          xInd,
               const float*        beta,
               float*              y,
               cusparseIndexBase_t idxBase,
               void*               pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgemvi(cusparseHandle_t    handle,
               cusparseOperation_t transA,
               int                 m,
               int                 n,
               const double*       alpha,
               const double*       A,
               int                 lda,
               int                 nnz,
               const double*       xVal,
               const int*          xInd,
               const double*       beta,
               double*             y,
               cusparseIndexBase_t idxBase,
               void*               pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgemvi(cusparseHandle_t    handle,
               cusparseOperation_t transA,
               int                 m,
               int                 n,
               const cuComplex*    alpha,
               const cuComplex*    A,
               int                 lda,
               int                 nnz,
               const cuComplex*    xVal,
               const int*          xInd,
               const cuComplex*    beta,
               cuComplex*          y,
               cusparseIndexBase_t idxBase,
               void*               pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgemvi(cusparseHandle_t       handle,
               cusparseOperation_t    transA,
               int                    m,
               int                    n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int                    lda,
               int                    nnz,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               const cuDoubleComplex* beta,
               cuDoubleComplex*       y,
               cusparseIndexBase_t    idxBase,
               void*                  pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgemvi_bufferSize(cusparseHandle_t    handle,
                          cusparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx_bufferSize(cusparseHandle_t         handle,
                           cusparseAlgMode_t        alg,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      n,
                           int                      nnz,
                           const void*              alpha,
                           cudaDataType             alphatype,
                           const cusparseMatDescr_t descrA,
                           const void*              csrValA,
                           cudaDataType             csrValAtype,
                           const int*               csrRowPtrA,
                           const int*               csrColIndA,
                           const void*              x,
                           cudaDataType             xtype,
                           const void*              beta,
                           cudaDataType             betatype,
                           void*                    y,
                           cudaDataType             ytype,
                           cudaDataType             executiontype,
                           size_t*                  bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx(cusparseHandle_t         handle,
                cusparseAlgMode_t        alg,
                cusparseOperation_t      transA,
                int                      m,
                int                      n,
                int                      nnz,
                const void*              alpha,
                cudaDataType             alphatype,
                const cusparseMatDescr_t descrA,
                const void*              csrValA,
                cudaDataType             csrValAtype,
                const int*               csrRowPtrA,
                const int*               csrColIndA,
                const void*              x,
                cudaDataType             xtype,
                const void*              beta,
                cudaDataType             betatype,
                void*                    y,
                cudaDataType             ytype,
                cudaDataType             executiontype,
                void*                    buffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseShybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const float*             x,
               const float*             beta,
               float*                   y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDhybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const double*            x,
               const double*            beta,
               double*                  y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseChybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZhybmv(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cusparseHybMat_t   hybA,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const float*             x,
               const float*             beta,
               float*                   y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double*            bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const double*            x,
               const double*            beta,
               double*                  y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex*         bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrmv(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex*   bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrxmv(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const float*             alpha,
                const cusparseMatDescr_t descrA,
                const float*             bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const float*             x,
                const float*             beta,
                float*                   y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrxmv(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const double*            alpha,
                const cusparseMatDescr_t descrA,
                const double*            bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const double*            x,
                const double*            beta,
                double*                  y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrxmv(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const cuComplex*         alpha,
                const cusparseMatDescr_t descrA,
                const cuComplex*         bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const cuComplex*         x,
                const cuComplex*         beta,
                cuComplex*               y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrxmv(cusparseHandle_t      handle,
             cusparseDirection_t      dirA,
             cusparseOperation_t      transA,
             int                      sizeOfMask,
             int                      mb,
             int                      nb,
             int                      nnzb,
             const cuDoubleComplex*   alpha,
             const cusparseMatDescr_t descrA,
             const cuDoubleComplex*   bsrSortedValA,
             const int*               bsrSortedMaskPtrA,
             const int*               bsrSortedRowPtrA,
             const int*               bsrSortedEndPtrA,
             const int*               bsrSortedColIndA,
             int                      blockDim,
             const cuDoubleComplex*   x,
             const cuDoubleComplex*   beta,
             cuDoubleComplex*         y)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                          csrsv2Info_t     info,
                          int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           float*                   csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           double*                  csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              float*                   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              double*                  csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              cuComplex*               csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const double*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const float*             f,
                      float*                   x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const double*            f,
                      double*                  x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const cuComplex*         f,
                      cuComplex*               x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const cuDoubleComplex*   f,
                      cuDoubleComplex*         x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle,
                          bsrsv2Info_t     info,
                          int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           float*                   bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           double*                  bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           bsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              float*                   bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              double*                  bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuComplex*               bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex*         bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              bsrsv2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const float*             bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const double*            bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         bsrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const float*             f,
                      float*                   x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const double*            f,
                      double*                  x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const cuComplex*         f,
                      cuComplex*               x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      bsrsv2Info_t             info,
                      const cuDoubleComplex*   f,
                      cuDoubleComplex*         x,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseShybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDhybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseChybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZhybsv_analysis(cusparseHandle_t            handle,
                        cusparseOperation_t         transA,
                        const cusparseMatDescr_t    descrA,
                        cusparseHybMat_t            hybA,
                        cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseShybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const float*                alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const float*                f,
                     float*                      x)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseChybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const cuComplex*            alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuComplex*            f,
                     cuComplex*                  x)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDhybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const double*               alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const double*               f,
                     double*                     x)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZhybsv_solve(cusparseHandle_t            handle,
                     cusparseOperation_t         trans,
                     const cuDoubleComplex*      alpha,
                     const cusparseMatDescr_t    descrA,
                     const cusparseHybMat_t      hybA,
                     cusparseSolveAnalysisInfo_t info,
                     const cuDoubleComplex*      f,
                     cuDoubleComplex*            x)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE LEVEL 3 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSbsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float* bsrSortedValA,
               const int*   bsrSortedRowPtrA,
               const int*   bsrSortedColIndA,
               const int    blockSize,
               const float* B,
               const int    ldb,
               const float* beta,
               float*       C,
               int          ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double* bsrSortedValA,
               const int*    bsrSortedRowPtrA,
               const int*    bsrSortedColIndA,
               const int     blockSize,
               const double* B,
               const int     ldb,
               const double* beta,
               double*       C,
               int           ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex* bsrSortedValA,
               const int*       bsrSortedRowPtrA,
               const int*       bsrSortedColIndA,
               const int        blockSize,
               const cuComplex* B,
               const int        ldb,
               const cuComplex* beta,
               cuComplex*       C,
               int              ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
 cusparseZbsrmm(cusparseHandle_t         handle,
                cusparseDirection_t      dirA,
                cusparseOperation_t      transA,
                cusparseOperation_t      transB,
                int                      mb,
                int                      n,
                int                      kb,
                int                      nnzb,
                const cuDoubleComplex*   alpha,
                const cusparseMatDescr_t descrA,
                const cuDoubleComplex*   bsrSortedValA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedColIndA,
                const int                blockSize,
                const cuDoubleComplex*   B,
                const int                ldb,
                const cuDoubleComplex*   beta,
                cuDoubleComplex*         C,
                int                      ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
                 cusparseSgemmi(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              k,
                                int              nnz,
                                const float*     alpha,
                                const float*     A,
                                int              lda,
                                const float*     cscValB,
                                const int*       cscColPtrB,
                                const int*       cscRowIndB,
                                const float*     beta,
                                float*           C,
                                int              ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
                 cusparseDgemmi(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              k,
                                int              nnz,
                                const double*    alpha,
                                const double*    A,
                                int              lda,
                                const double*    cscValB,
                                const int*       cscColPtrB,
                                const int*       cscRowIndB,
                                const double*    beta,
                                double*          C,
                                int              ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
                 cusparseCgemmi(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              k,
                                int              nnz,
                                const cuComplex* alpha,
                                const cuComplex* A,
                                int              lda,
                                const cuComplex* cscValB,
                                const int*       cscColPtrB,
                                const int*       cscRowIndB,
                                const cuComplex* beta,
                                cuComplex*       C,
                                int              ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
                 cusparseZgemmi(cusparseHandle_t       handle,
                                int                    m,
                                int                    n,
                                int                    k,
                                int                    nnz,
                                const cuDoubleComplex* alpha,
                                const cuDoubleComplex* A,
                                int                    lda,
                                const cuDoubleComplex* cscValB,
                                const int*             cscColPtrB,
                                const int*             cscRowIndB,
                                const cuDoubleComplex* beta,
                                cuDoubleComplex*       C,
                                int                    ldc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrsm2Info(csrsm2Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrsm2Info(csrsm2Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                          csrsm2Info_t     info,
                          int* position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const float*             alpha,
                              const cusparseMatDescr_t descrA,
                              const float*             csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const float*             B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const double*            alpha,
                              const cusparseMatDescr_t descrA,
                              const double*            csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const double*            B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const cuComplex*         alpha,
                              const cusparseMatDescr_t descrA,
                              const cuComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const cuComplex*         B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              int                      algo,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const cuDoubleComplex*   alpha,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex*   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const cuDoubleComplex*   B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              cusparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const float*             alpha,
                         const cusparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const float*             B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const double*            alpha,
                         const cusparseMatDescr_t descrA,
                         const double*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const double*            B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const cuComplex*         alpha,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const cuComplex*         B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_analysis(cusparseHandle_t         handle,
                         int                      algo,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const cuDoubleComplex*   alpha,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const cuDoubleComplex*   B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      float*                   B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      double*                  B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      cuComplex*               B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_solve(cusparseHandle_t         handle,
                      int                      algo,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      cuDoubleComplex*         B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle,
                          bsrsm2Info_t     info,
                          int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           float*                   bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           double*                  bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_bufferSize(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           cusparseOperation_t      transA,
                           cusparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              float*                   bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              double*                  bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuComplex*               bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              cusparseOperation_t      transA,
                              cusparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex*         bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const float*             bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const double*            bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuComplex*         bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_analysis(cusparseHandle_t         handle,
                         cusparseDirection_t      dirA,
                         cusparseOperation_t      transA,
                         cusparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex*   bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         bsrsm2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const float*             alpha,
                      const cusparseMatDescr_t descrA,
                      const float*             bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const float*             B,
                      int                      ldb,
                      float*                   X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const double*            alpha,
                      const cusparseMatDescr_t descrA,
                      const double*            bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const double*            B,
                      int                      ldb,
                      double*                  X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const cuComplex*         alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex*         bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const cuComplex*         B,
                      int                      ldb,
                      cuComplex*               X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_solve(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      cusparseOperation_t      transA,
                      cusparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const cuDoubleComplex*   alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex*   bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      bsrsm2Info_t             info,
                      const cuDoubleComplex*   B,
                      int                      ldb,
                      cuDoubleComplex*         X,
                      int                      ldx,
                      cusparseSolvePolicy_t    policy,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# PRECONDITIONERS
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCsrilu0Ex(cusparseHandle_t            handle,
                  cusparseOperation_t         trans,
                  int                         m,
                  const cusparseMatDescr_t    descrA,
                  void*                       csrSortedValA_ValM,
                  cudaDataType                csrSortedValA_ValMtype,
                  const int*                  csrSortedRowPtrA,
                  const int*                  csrSortedColIndA,
                  cusparseSolveAnalysisInfo_t info,
                  cudaDataType                executiontype)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrilu0(cusparseHandle_t           handle,
                cusparseOperation_t         trans,
                int                         m,
                const cusparseMatDescr_t    descrA,
                float*                      csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu0(cusparseHandle_t            handle,
                 cusparseOperation_t         trans,
                 int                         m,
                 const cusparseMatDescr_t    descrA,
                 double*                     csrSortedValA_ValM,
                 const int*                  csrSortedRowPtrA,
                 const int*                  csrSortedColIndA,
                 cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu0(cusparseHandle_t         handle,
              cusparseOperation_t         trans,
              int                         m,
              const cusparseMatDescr_t    descrA,
              cuComplex*                  csrSortedValA_ValM,
              const int*                  csrSortedRowPtrA,
              const int*                  csrSortedColIndA,
              cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu0(cusparseHandle_t            handle,
                 cusparseOperation_t         trans,
                 int                         m,
                 const cusparseMatDescr_t    descrA,
                 cuDoubleComplex*            csrSortedValA_ValM,
                 const int*                  csrSortedRowPtrA,
                 const int*                  csrSortedColIndA,
                 cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle,
                            csrilu02Info_t   info,
                            int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             float*                   csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             double*                  csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             cuComplex*               csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             cuDoubleComplex*         csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                float*                   csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                double*                  csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                cuComplex*               csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                cuDoubleComplex*         csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const float*             csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const double*            csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const cuComplex*         csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const cuDoubleComplex*   csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                   csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  csrilu02Info_t        info,
                  cusparseSolvePolicy_t policy,
                  void*                 pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                            bsrilu02Info_t   info,
                            int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             float*                   bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             double*                  bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             cuComplex*               bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_bufferSize(cusparseHandle_t         handle,
                             cusparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const cusparseMatDescr_t descrA,
                             cuDoubleComplex*         bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                float*                   bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                double*                  bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                cuComplex*               bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex*         bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsrilu02Info_t           info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           float*                   bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           double*                  bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex*               bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_analysis(cusparseHandle_t         handle,
                           cusparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           bsrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  float*                   bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  double*                  bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02(cusparseHandle_t         handle,
                  cusparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  bsrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsric0(cusparseHandle_t            handle,
                cusparseOperation_t         trans,
                int                         m,
                const cusparseMatDescr_t    descrA,
                float*                      csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsric0(cusparseHandle_t         handle,
                cusparseOperation_t      trans,
                int                      m,
                const cusparseMatDescr_t descrA,
                double*                  csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsric0(cusparseHandle_t         handle,
                cusparseOperation_t      trans,
                int                      m,
                const cusparseMatDescr_t descrA,
                cuComplex*               csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsric0(cusparseHandle_t            handle,
                cusparseOperation_t         trans,
                int                         m,
                const cusparseMatDescr_t    descrA,
                cuDoubleComplex*            csrSortedValA_ValM,
                const int*                  csrSortedRowPtrA,
                const int*                  csrSortedColIndA,
                cusparseSolveAnalysisInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                           csric02Info_t    info,
                           int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            float*                   csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            double*                  csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            cuComplex*               csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            cuDoubleComplex*         csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               float*                   csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               double*                  csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               cuComplex*               csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_bufferSizeExt(cusparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex*         csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex*   csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 float*                   csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 double*                  csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 cuComplex*               csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 cuDoubleComplex*         csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                           bsric02Info_t    info,
                           int*             position)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            float*                   bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            double*                  bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            cuComplex*               bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_bufferSize(cusparseHandle_t         handle,
                            cusparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const cusparseMatDescr_t descrA,
                            cuDoubleComplex*         bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               float*                   bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
 cusparseDbsric02_bufferSizeExt(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                double*                  bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsric02Info_t            info,
                                size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               cuComplex*               bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_bufferSizeExt(cusparseHandle_t         handle,
                               cusparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex*         bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const float*             bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const double*            bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_analysis(cusparseHandle_t         handle,
                          cusparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex*   bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          bsric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void*                    pInputBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 float*                   bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 double*                  bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 cuComplex*               bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*
                      bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsric02(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const cusparseMatDescr_t descrA,
                 cuDoubleComplex*         bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 bsric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv(cusparseHandle_t handle,
              int              m,
              int              n,
              const float*     dl,
              const float*     d,
              const float*     du,
              float*           B,
              int              ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv(cusparseHandle_t handle,
              int              m,
              int              n,
              const double*    dl,
              const double*    d,
              const double*    du,
              double*          B,
              int              ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv(cusparseHandle_t handle,
              int              m,
              int              n,
              const cuComplex* dl,
              const cuComplex* d,
              const cuComplex* du,
              cuComplex*       B,
              int              ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv(cusparseHandle_t       handle,
              int                    m,
              int                    n,
              const cuDoubleComplex* dl,
              const cuDoubleComplex* d,
              const cuDoubleComplex* du,
              cuDoubleComplex*       B,
              int                    ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int              m,
                             int              n,
                             const float*     dl,
                             const float*     d,
                             const float*     du,
                             const float*     B,
                             int              ldb,
                             size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int              m,
                             int              n,
                             const double*    dl,
                             const double*    d,
                             const double*    du,
                             const double*    B,
                             int              ldb,
                             size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int              m,
                             int              n,
                             const cuComplex* dl,
                             const cuComplex* d,
                             const cuComplex* du,
                             const cuComplex* B,
                             int              ldb,
                             size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_bufferSizeExt(cusparseHandle_t       handle,
                             int                    m,
                             int                    n,
                             const cuDoubleComplex* dl,
                             const cuDoubleComplex* d,
                             const cuDoubleComplex* du,
                             const cuDoubleComplex* B,
                             int                    ldb,
                             size_t*                bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2(cusparseHandle_t handle,
               int              m,
               int              n,
               const float*     dl,
               const float*     d,
               const float*     du,
               float*           B,
               int              ldb,
               void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2(cusparseHandle_t handle,
               int              m,
               int              n,
               const double*    dl,
               const double*    d,
               const double*    du,
               double*          B,
               int              ldb,
               void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2(cusparseHandle_t handle,
               int              m,
               int              n,
               const cuComplex* dl,
               const cuComplex* d,
               const cuComplex* du,
               cuComplex*       B,
               int              ldb,
               void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2(cusparseHandle_t       handle,
               int                    m,
               int                    n,
               const cuDoubleComplex* dl,
               const cuDoubleComplex* d,
               const cuDoubleComplex* du,
               cuDoubleComplex*       B,
               int                    ldb,
               void*                  pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const float*     dl,
                      const float*     d,
                      const float*     du,
                      float*           B,
                      int              ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv_nopivot(cusparseHandle_t handle,
                                                   int              m,
                                                   int              n,
                                                   const double*    dl,
                                                   const double*    d,
                                                   const double*    du,
                                                   double*          B,
                                                   int              ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const cuComplex* dl,
                      const cuComplex* d,
                      const cuComplex* du,
                      cuComplex*       B,
                      int              ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv_nopivot(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      const cuDoubleComplex* dl,
                      const cuDoubleComplex* d,
                      const cuDoubleComplex* du,
                      cuDoubleComplex*       B,
                      int                    ldb)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const float*     dl,
                                     const float*     d,
                                     const float*     du,
                                     const float*     B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const double*    dl,
                                     const double*    d,
                                     const double*    du,
                                     const double*    B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const cuComplex* dl,
                                     const cuComplex* d,
                                     const cuComplex* du,
                                     const cuComplex* B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t       handle,
                                     int                    m,
                                     int                    n,
                                     const cuDoubleComplex* dl,
                                     const cuDoubleComplex* d,
                                     const cuDoubleComplex* du,
                                     const cuDoubleComplex* B,
                                     int                    ldb,
                                     size_t*                bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_nopivot(cusparseHandle_t handle,
                       int              m,
                       int              n,
                       const float*     dl,
                       const float*     d,
                       const float*     du,
                       float*           B,
                       int              ldb,
                       void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_nopivot(cusparseHandle_t handle,
                       int              m,
                       int              n,
                       const double*    dl,
                       const double*    d,
                       const double*    du,
                       double*          B,
                       int              ldb,
                       void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_nopivot(cusparseHandle_t handle,
                       int              m,
                       int              n,
                       const cuComplex* dl,
                       const cuComplex* d,
                       const cuComplex* du,
                       cuComplex*       B,
                       int              ldb,
                       void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_nopivot(cusparseHandle_t       handle,
                       int                    m,
                       int                    n,
                       const cuDoubleComplex* dl,
                       const cuDoubleComplex* d,
                       const cuDoubleComplex* du,
                       cuDoubleComplex*       B,
                       int                    ldb,
                       void*                  pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsvStridedBatch(cusparseHandle_t handle,
                          int              m,
                          const float*     dl,
                          const float*     d,
                          const float*     du,
                          float*           x,
                          int              batchCount,
                          int              batchStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsvStridedBatch(cusparseHandle_t handle,
                          int              m,
                          const double*    dl,
                          const double*    d,
                          const double*    du,
                          double*          x,
                          int              batchCount,
                          int              batchStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsvStridedBatch(cusparseHandle_t handle,
                          int              m,
                          const cuComplex* dl,
                          const cuComplex* d,
                          const cuComplex* du,
                          cuComplex*       x,
                          int              batchCount,
                          int              batchStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsvStridedBatch(cusparseHandle_t       handle,
                          int                    m,
                          const cuDoubleComplex* dl,
                          const cuDoubleComplex* d,
                          const cuDoubleComplex* du,
                          cuDoubleComplex*       x,
                          int                    batchCount,
                          int                    batchStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              m,
                                         const float*     dl,
                                         const float*     d,
                                         const float*     du,
                                         const float*     x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              m,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         const double*    x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              m,
                                         const cuComplex* dl,
                                         const cuComplex* d,
                                         const cuComplex* du,
                                         const cuComplex* x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
 cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t       handle,
                                          int                    m,
                                          const cuDoubleComplex* dl,
                                          const cuDoubleComplex* d,
                                          const cuDoubleComplex* du,
                                          const cuDoubleComplex* x,
                                          int                    batchCount,
                                          int                    batchStride,
                                          size_t* bufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2StridedBatch(cusparseHandle_t handle,
                           int              m,
                           const float*     dl,
                           const float*     d,
                           const float*     du,
                           float*           x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch(cusparseHandle_t handle,
                           int              m,
                           const double*    dl,
                           const double*    d,
                           const double*    du,
                           double*          x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2StridedBatch(cusparseHandle_t handle,
                           int              m,
                           const cuComplex* dl,
                           const cuComplex* d,
                           const cuComplex* du,
                           cuComplex*       x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2StridedBatch(cusparseHandle_t       handle,
                           int                    m,
                           const cuDoubleComplex* dl,
                           const cuDoubleComplex* d,
                           const cuDoubleComplex* du,
                           cuDoubleComplex*       x,
                           int                    batchCount,
                           int                    batchStride,
                           void*                  pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int              algo,
                                         int              m,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         const double*    x,
                                         int              batchCount,
                                         size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const cuComplex* dl,
                                            const cuComplex* d,
                                            const cuComplex* du,
                                            const cuComplex* x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
 cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t       handle,
                                             int                    algo,
                                             int                    m,
                                             const cuDoubleComplex* dl,
                                             const cuDoubleComplex* d,
                                             const cuDoubleComplex* du,
                                             const cuDoubleComplex* x,
                                             int                    batchCount,
                                             size_t*        pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              float*           dl,
                              float*           d,
                              float*           du,
                              float*           x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              double*          dl,
                              double*          d,
                              double*          du,
                              double*          x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuComplex*       dl,
                              cuComplex*       d,
                              cuComplex*       du,
                              cuComplex*       x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgtsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuDoubleComplex* dl,
                              cuDoubleComplex* d,
                              cuDoubleComplex* du,
                              cuDoubleComplex* x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const float*     ds,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     dw,
                                            const float*     x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const double*    ds,
                                            const double*    dl,
                                            const double*    d,
                                            const double*    du,
                                            const double*    dw,
                                            const double*    x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const cuComplex* ds,
                                            const cuComplex* dl,
                                            const cuComplex* d,
                                            const cuComplex* du,
                                            const cuComplex* dw,
                                            const cuComplex* x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t       handle,
                                            int                    algo,
                                            int                    m,
                                            const cuDoubleComplex* ds,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* dw,
                                            const cuDoubleComplex* x,
                                            int                    batchCount,
                                            size_t*         pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              float*           ds,
                              float*           dl,
                              float*           d,
                              float*           du,
                              float*           dw,
                              float*           x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              double*          ds,
                              double*          dl,
                              double*          d,
                              double*          du,
                              double*          dw,
                              double*          x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuComplex*       ds,
                              cuComplex*       dl,
                              cuComplex*       d,
                              cuComplex*       du,
                              cuComplex*       dw,
                              cuComplex*       x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgpsvInterleavedBatch(cusparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuDoubleComplex* ds,
                              cuDoubleComplex* dl,
                              cuDoubleComplex* d,
                              cuDoubleComplex* du,
                              cuDoubleComplex* dw,
                              cuDoubleComplex* x,
                              int              batchCount,
                              void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE LEVEL 4 ROUTINES #
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrgemm2Info(csrgemm2Info_t* info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const float*             beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const double*            beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const cuComplex*         alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cuComplex*         beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const cuDoubleComplex*   alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cuDoubleComplex*   beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrgemm2Nnz(cusparseHandle_t         handle,
                     int                      m,
                     int                      n,
                     int                      k,
                     const cusparseMatDescr_t descrA,
                     int                      nnzA,
                     const int*               csrSortedRowPtrA,
                     const int*               csrSortedColIndA,
                     const cusparseMatDescr_t descrB,
                     int                      nnzB,
                     const int*               csrSortedRowPtrB,
                     const int*               csrSortedColIndB,
                     const cusparseMatDescr_t descrD,
                     int                      nnzD,
                     const int*               csrSortedRowPtrD,
                     const int*               csrSortedColIndD,
                     const cusparseMatDescr_t descrC,
                     int*                     csrSortedRowPtrC,
                     int*                     nnzTotalDevHostPtr,
                     const csrgemm2Info_t     info,
                     void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const float*             alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const float*             csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const float*             beta,
                  const cusparseMatDescr_t descrD,
                  int                      nnzD,
                  const float*             csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  float*                   csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const double*            alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const double*            csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const double*            beta,
                  const cusparseMatDescr_t descrD,
                  int                      nnzD,
                  const double*            csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  double*                  csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm2(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cuComplex*         alpha,
                 const cusparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const cusparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuComplex*         csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cuComplex*         beta,
                 const cusparseMatDescr_t descrD,
                 int                      nnzD,
                 const cuComplex*         csrSortedValD,
                 const int*               csrSortedRowPtrD,
                 const int*               csrSortedColIndD,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC,
                 const csrgemm2Info_t     info,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const cuDoubleComplex*   alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuDoubleComplex*   csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cuDoubleComplex*   beta,
                  const cusparseMatDescr_t descrD,
                  int                      nnzD,
                  const cuDoubleComplex*   csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex*         csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const float*             csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const float*             beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const float*             csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const float*             csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const double*            csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const double*            beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const double*            csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const double*            csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const cuComplex*         alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const cuComplex*         csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuComplex*         beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const cuComplex*         csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const cuComplex*         csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const cuDoubleComplex*   alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const cuDoubleComplex*   csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuDoubleComplex*   beta,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const cuDoubleComplex*   csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const cuDoubleComplex*   csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrgeam2Nnz(cusparseHandle_t         handle,
                     int                      m,
                     int                      n,
                     const cusparseMatDescr_t descrA,
                     int                      nnzA,
                     const int*               csrSortedRowPtrA,
                     const int*               csrSortedColIndA,
                     const cusparseMatDescr_t descrB,
                     int                      nnzB,
                     const int*               csrSortedRowPtrB,
                     const int*               csrSortedColIndB,
                     const cusparseMatDescr_t descrC,
                     int*                     csrSortedRowPtrC,
                     int*                     nnzTotalDevHostPtr,
                     void*                    workspace)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const float*             alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const float*             beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const float*             csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  float*                   csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const double*            alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const double*            beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const double*            csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  double*                  csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const cuComplex*         alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuComplex*         csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuComplex*         beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuComplex*         csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuComplex*               csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam2(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const cuDoubleComplex*   alpha,
                  const cusparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuDoubleComplex*   beta,
                  const cusparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuDoubleComplex*   csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex*         csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


/* --- Sparse Matrix Reorderings --- */

/* Description: Find an approximate coloring of a matrix stored in CSR format.
 */
cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(cusparseHandle_t         handle,
                                               int                      m,
                                               int                      nnz,
                                               const cusparseMatDescr_t descrA,
                                               const float* csrSortedValA,
                                               const int*   csrSortedRowPtrA,
                                               const int*   csrSortedColIndA,
                                               const float* fractionToColor,
                                               int*         ncolors,
                                               int*         coloring,
                                               int*         reordering,
                                               const cusparseColorInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(cusparseHandle_t         handle,
                                               int                      m,
                                               int                      nnz,
                                               const cusparseMatDescr_t descrA,
                                               const double* csrSortedValA,
                                               const int*    csrSortedRowPtrA,
                                               const int*    csrSortedColIndA,
                                               const double* fractionToColor,
                                               int*          ncolors,
                                               int*          coloring,
                                               int*          reordering,
                                               const cusparseColorInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(cusparseHandle_t         handle,
                                               int                      m,
                                               int                      nnz,
                                               const cusparseMatDescr_t descrA,
                                               const cuComplex* csrSortedValA,
                                               const int*   csrSortedRowPtrA,
                                               const int*   csrSortedColIndA,
                                               const float* fractionToColor,
                                               int*         ncolors,
                                               int*         coloring,
                                               int*         reordering,
                                               const cusparseColorInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
                 cusparseZcsrcolor(cusparseHandle_t          handle,
                                   int                       m,
                                   int                       nnz,
                                   const cusparseMatDescr_t  descrA,
                                   const cuDoubleComplex*    csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const double*             fractionToColor,
                                   int*                      ncolors,
                                   int*                      coloring,
                                   int*                      reordering,
                                   const cusparseColorInfo_t info)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE FORMAT CONVERSION
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const float*             A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const double*            A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const cuComplex*         A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZnnz(cusparseHandle_t         handle,
             cusparseDirection_t      dirA,
             int                      m,
             int                      n,
             const cusparseMatDescr_t descrA,
             const cuDoubleComplex*   A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE FORMAT CONVERSION #
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      float                    tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      double                   tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      cuComplex                tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZnnz_compress(cusparseHandle_t         handle,
                      int                      m,
                      const cusparseMatDescr_t descr,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      cuDoubleComplex          tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          float*                   csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          float                    tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          double*                  csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          double                   tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuComplex*               csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuComplex                tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex*   csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuDoubleComplex*         csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuDoubleComplex          tol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSdense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   float*                   csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDdense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerRow,
                   double*                  csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCdense2csr(cusparseHandle_t           handle,
                     int                      m,
                     int                      n,
                     const cusparseMatDescr_t descrA,
                     const cuComplex*         A,
                     int                      lda,
                     const int*               nnzPerRow,
                     cuComplex*               csrSortedValA,
                     int*                     csrSortedRowPtrA,
                     int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZdense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cuDoubleComplex*         csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   float*                   A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   double*                  A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   cuComplex*               A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2dense(cusparseHandle_t         handle,
                int                      m,
                int                      n,
                const cusparseMatDescr_t descrA,
                const cuDoubleComplex*   csrSortedValA,
                const int*               csrSortedRowPtrA,
                const int*               csrSortedColIndA,
                cuDoubleComplex*         A,
                int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerCol,
                   float*                   cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerCol,
                   double*                  cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         A,
                   int                      lda,
                   const int*               nnzPerCol,
                   cuComplex*               cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZdense2csc(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerCol,
                   cuDoubleComplex*         cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   float*                   A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   double*                  A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   cuComplex*               A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsc2dense(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   cuDoubleComplex*         A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcoo2csr(cusparseHandle_t    handle,
                 const int*          cooRowInd,
                 int                 nnz,
                 int                 m,
                 int*                csrSortedRowPtr,
                 cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsr2coo(cusparseHandle_t    handle,
                 const int*          csrSortedRowPtr,
                 int                 nnz,
                 int                 m,
                 int*                cooRowInd,
                 cusparseIndexBase_t idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZdense2hyb(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cusparseHybMat_t         hybA,
                   int                      userEllWidth,
                   cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseShyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   float*                   A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDhyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   double*                  A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseChyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   cuComplex*               A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZhyb2dense(cusparseHandle_t         handle,
                   const cusparseMatDescr_t descrA,
                   const cusparseHybMat_t   hybA,
                   cuDoubleComplex*         A,
                   int                      lda)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseShyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 float*                   csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDhyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 double*                  csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseChyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuComplex*               csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZhyb2csr(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuDoubleComplex*         csrSortedValA,
                 int*                     csrSortedRowPtrA,
                 int*                     csrSortedColIndA)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsc2hyb(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   cscSortedValA,
                 const int*               cscSortedRowIndA,
                 const int*               cscSortedColPtrA,
                 cusparseHybMat_t         hybA,
                 int                      userEllWidth,
                 cusparseHybPartition_t   partitionType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseShyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 float*                   cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDhyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 double*                  cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseChyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuComplex*               cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZhyb2csc(cusparseHandle_t         handle,
                 const cusparseMatDescr_t descrA,
                 const cusparseHybMat_t   hybA,
                 cuDoubleComplex*         cscSortedVal,
                 int*                     cscSortedRowInd,
                 int*                     cscSortedColPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsr2bsrNnz(cusparseHandle_t         handle,
                    cusparseDirection_t      dirA,
                    int                      m,
                    int                      n,
                    const cusparseMatDescr_t descrA,
                    const int*               csrSortedRowPtrA,
                    const int*               csrSortedColIndA,
                    int                      blockDim,
                    const cusparseMatDescr_t descrC,
                    int*                     bsrSortedRowPtrC,
                    int*                     nnzTotalDevHostPtr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 float*                   bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 double*                  bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex*         bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const float*             bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 float*                   csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const double*            bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 double*                  csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const cuComplex*         bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZbsr2csr(cusparseHandle_t         handle,
                 cusparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex*   bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex*         csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const float*     bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const double*    bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const cuComplex* bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t       handle,
                                int                    mb,
                                int                    nb,
                                int                    nnzb,
                                const cuDoubleComplex* bsrSortedVal,
                                const int*             bsrSortedRowPtr,
                                const int*             bsrSortedColInd,
                                int                    rowBlockDim,
                                int                    colBlockDim,
                                int*                   pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const float*     bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const double*    bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const cuComplex* bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t       handle,
                                   int                    mb,
                                   int                    nb,
                                   int                    nnzb,
                                   const cuDoubleComplex* bsrSortedVal,
                                   const int*             bsrSortedRowPtr,
                                   const int*             bsrSortedColInd,
                                   int                    rowBlockDim,
                                   int                    colBlockDim,
                                   size_t*                pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc(cusparseHandle_t handle,
                     int              mb,
                     int              nb,
                     int              nnzb,
                     const float*     bsrSortedVal,
                     const int* bsrSortedRowPtr,
                     const int* bsrSortedColInd,
                     int        rowBlockDim,
                     int        colBlockDim,
                     float*     bscVal,
                     int*       bscRowInd,
                     int*       bscColPtr,
                     cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase,
                     void*               pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc(cusparseHandle_t    handle,
                     int                 mb,
                     int                 nb,
                     int                 nnzb,
                     const double*       bsrSortedVal,
                     const int*          bsrSortedRowPtr,
                     const int*          bsrSortedColInd,
                     int                 rowBlockDim,
                     int                 colBlockDim,
                     double*             bscVal,
                     int*                bscRowInd,
                     int*                bscColPtr,
                     cusparseAction_t    copyValues,
                     cusparseIndexBase_t idxBase,
                     void*               pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc(cusparseHandle_t    handle,
                     int                 mb,
                     int                 nb,
                     int                 nnzb,
                     const cuComplex*    bsrSortedVal,
                     const int*          bsrSortedRowPtr,
                     const int*          bsrSortedColInd,
                     int                 rowBlockDim,
                     int                 colBlockDim,
                     cuComplex*          bscVal,
                     int*                bscRowInd,
                     int*                bscColPtr,
                     cusparseAction_t    copyValues,
                     cusparseIndexBase_t idxBase,
                     void*               pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc(cusparseHandle_t       handle,
                     int                    mb,
                     int                    nb,
                     int                    nnzb,
                     const cuDoubleComplex* bsrSortedVal,
                     const int*             bsrSortedRowPtr,
                     const int*             bsrSortedColInd,
                     int                    rowBlockDim,
                     int                    colBlockDim,
                     cuDoubleComplex*       bscVal,
                     int*                   bscRowInd,
                     int*                   bscColPtr,
                     cusparseAction_t       copyValues,
                     cusparseIndexBase_t    idxBase,
                     void*                  pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const float*             bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   float*                   csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const double*            bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   double*                  csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   cuComplex*               csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2csr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const cusparseMatDescr_t descrC,
                   cuDoubleComplex*         csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const float*             csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const double*            csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const cuComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr_bufferSize(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex*   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const float*             csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const double*            csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const cuComplex*         csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                 cusparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const cuDoubleComplex*   csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsr2gebsrNnz(cusparseHandle_t         handle,
                      cusparseDirection_t      dirA,
                      int                      m,
                      int                      n,
                      const cusparseMatDescr_t descrA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const cusparseMatDescr_t descrC,
                      int*                     bsrSortedRowPtrC,
                      int                      rowBlockDim,
                      int                      colBlockDim,
                      int*                     nnzTotalDevHostPtr,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   float*                   bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*            csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   double*                  bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex*         csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   cuComplex*               bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr(cusparseHandle_t         handle,
                   cusparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex*   csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   cuDoubleComplex*         bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const float*             bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const double*            bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const cuComplex*         bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t         handle,
                                cusparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const cusparseMatDescr_t descrA,
                                const cuDoubleComplex*   bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const float*             bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const double*            bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const cuComplex*         bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr_bufferSizeExt(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const cuDoubleComplex*   bsrSortedValA,
                                   const int*               bsrSortedRowPtrA,
                                   const int*               bsrSortedColIndA,
                                   int                      rowBlockDimA,
                                   int                      colBlockDimA,
                                   int                      rowBlockDimC,
                                   int                      colBlockDimC,
                                   size_t*                  pBufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXgebsr2gebsrNnz(cusparseHandle_t         handle,
                        cusparseDirection_t      dirA,
                        int                      mb,
                        int                      nb,
                        int                      nnzb,
                        const cusparseMatDescr_t descrA,
                        const int*               bsrSortedRowPtrA,
                        const int*               bsrSortedColIndA,
                        int                      rowBlockDimA,
                        int                      colBlockDimA,
                        const cusparseMatDescr_t descrC,
                        int*                     bsrSortedRowPtrC,
                        int                      rowBlockDimC,
                        int                      colBlockDimC,
                        int*                     nnzTotalDevHostPtr,
                        void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const float*             bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     float*                   bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const double*            bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     double*                  bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const cuComplex*         bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     cuComplex*               bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr(cusparseHandle_t         handle,
                     cusparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const cusparseMatDescr_t descrA,
                     const cuDoubleComplex*   bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     cuDoubleComplex*         bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# SPARSE MATRIX SORTING
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateIdentityPermutation(cusparseHandle_t handle,
                                  int              n,
                                  int*             p)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       cooRowsA,
                               const int*       cooColsA,
                               size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcoosortByRow(cusparseHandle_t handle,
                      int              m,
                      int              n,
                      int              nnz,
                      int*             cooRowsA,
                      int*             cooColsA,
                      int*             P,
                      void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcoosortByColumn(cusparseHandle_t handle,
                         int              m,
                         int              n,
                         int              nnz,
                         int*             cooRowsA,
                         int*             cooColsA,
                         int*             P,
                         void*            pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       csrRowPtrA,
                               const int*       csrColIndA,
                               size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcsrsort(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 const int*               csrRowPtrA,
                 int*                     csrColIndA,
                 int*                     P,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       cscColPtrA,
                               const int*       cscRowIndA,
                               size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseXcscsort(cusparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 const int*               cscColPtrA,
                 int*                     cscRowIndA,
                 int*                     P,
                 void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                float*           csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                double*          csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuComplex*       csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuDoubleComplex* csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                   csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsru2csr(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseScsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                   csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex*               csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csru(cusparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex*         csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  csru2csrInfo_t           info,
                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csr_bufferSizeExt(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const __half*            A,
                                      int                      lda,
                                      const __half*            threshold,
                                      const cusparseMatDescr_t descrC,
                                      const __half*            csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t* pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csr_bufferSizeExt(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const float*             A,
                                      int                      lda,
                                      const float*             threshold,
                                      const cusparseMatDescr_t descrC,
                                      const float*             csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t* pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csr_bufferSizeExt(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const double*            A,
                                      int                      lda,
                                      const double*            threshold,
                                      const cusparseMatDescr_t descrC,
                                      const double*            csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t*               pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const __half*            A,
                           int                      lda,
                           const __half*            threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const float*             A,
                           int                      lda,
                           const float*             threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const double*            A,
                           int                      lda,
                           const double*            threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrSortedRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csr(cusparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const __half*            A,
                        int                      lda,
                        const __half*            threshold,
                        const cusparseMatDescr_t descrC,
                        __half*                  csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csr(cusparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const float*             A,
                        int                      lda,
                        const float*             threshold,
                        const cusparseMatDescr_t descrC,
                        float*                   csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csr(cusparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const double*            A,
                        int                      lda,
                        const double*            threshold,
                        const cusparseMatDescr_t descrC,
                        double*                  csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csr_bufferSizeExt(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const __half*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const __half*            threshold,
                                    const cusparseMatDescr_t descrC,
                                    const __half*            csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t* pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csr_bufferSizeExt(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const float*             threshold,
                                    const cusparseMatDescr_t descrC,
                                    const float*             csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t*                 pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csr_bufferSizeExt(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const double*            threshold,
                                    const cusparseMatDescr_t descrC,
                                    const double*            csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t*                 pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrNnz(cusparseHandle_t         handle,
                         int                      m,
                         int                      n,
                         int                      nnzA,
                         const cusparseMatDescr_t descrA,
                         const __half*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const __half*            threshold,
                         const cusparseMatDescr_t descrC,
                         int*                     csrSortedRowPtrC,
                         int*                     nnzTotalDevHostPtr,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrNnz(cusparseHandle_t         handle,
                         int                      m,
                         int                      n,
                         int                      nnzA,
                         const cusparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const float*             threshold,
                         const cusparseMatDescr_t descrC,
                         int*                     csrSortedRowPtrC,
                         int*                     nnzTotalDevHostPtr,
                         void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
 cusparseDpruneCsr2csrNnz(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          int                      nnzA,
                          const cusparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          const double*            threshold,
                          const cusparseMatDescr_t descrC,
                          int*                     csrSortedRowPtrC,
                          int*                     nnzTotalDevHostPtr,
                          void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csr(cusparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const cusparseMatDescr_t descrA,
                      const __half*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const __half*            threshold,
                      const cusparseMatDescr_t descrC,
                      __half*                  csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csr(cusparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const cusparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const float*             threshold,
                      const cusparseMatDescr_t descrC,
                      float*                   csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csr(cusparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const cusparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const double*            threshold,
                      const cusparseMatDescr_t descrC,
                      double*                  csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const __half*            A,
                                   int                      lda,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const __half*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const float*             A,
                                   int                      lda,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const float*             csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const double*            A,
                                   int                      lda,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const double*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const __half*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const float*             A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const double*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrByPercentage(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const __half*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    __half*                  csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrByPercentage(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const float*             A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    float*                   csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrByPercentage(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const double*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    double*                  csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const cusparseMatDescr_t descrA,
                                   const __half*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const __half*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const cusparseMatDescr_t descrA,
                                   const float*             csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const float*             csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
                                   cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const cusparseMatDescr_t descrA,
                                   const double*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const cusparseMatDescr_t descrC,
                                   const double*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const __half*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrNnzByPercentage(
                                    cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const cusparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrByPercentage(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const __half*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float percentage, /* between 0 to 100 */
                                  const cusparseMatDescr_t descrC,
                                  __half*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrByPercentage(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const float*             csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float                    percentage,
                                  const cusparseMatDescr_t descrC,
                                  float*                   csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrByPercentage(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const double*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float                    percentage,
                                  const cusparseMatDescr_t descrC,
                                  double*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//##############################################################################
//# CSR2CSC
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx2(cusparseHandle_t     handle,
                   int                  m,
                   int                  n,
                   int                  nnz,
                   const void*          csrVal,
                   const int*           csrRowPtr,
                   const int*           csrColInd,
                   void*                cscVal,
                   int*                 cscColPtr,
                   int*                 cscRowInd,
                   cudaDataType         valType,
                   cusparseAction_t     copyValues,
                   cusparseIndexBase_t  idxBase,
                   cusparseCsr2CscAlg_t alg,
                   void*                buffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx2_bufferSize(cusparseHandle_t     handle,
                              int                  m,
                              int                  n,
                              int                  nnz,
                              const void*          csrVal,
                              const int*           csrRowPtr,
                              const int*           csrColInd,
                              void*                cscVal,
                              int*                 cscColPtr,
                              int*                 cscRowInd,
                              cudaDataType         valType,
                              cusparseAction_t     copyValues,
                              cusparseIndexBase_t  idxBase,
                              cusparseCsr2CscAlg_t alg,
                              size_t*              bufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// SPARSE VECTOR DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr,
                    int64_t               size,
                    int64_t               nnz,
                    void*                 indices,
                    void*                 values,
                    cusparseIndexType_t   idxType,
                    cusparseIndexBase_t   idxBase,
                    cudaDataType          valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpVecGet(const cusparseSpVecDescr_t spVecDescr,
                 int64_t*                   size,
                 int64_t*                   nnz,
                 void**                     indices,
                 void**                     values,
                 cusparseIndexType_t*       idxType,
                 cusparseIndexBase_t*       idxBase,
                 cudaDataType*              valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetIndexBase(const cusparseSpVecDescr_t spVecDescr,
                          cusparseIndexBase_t*       idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetValues(const cusparseSpVecDescr_t spVecDescr,
                       void**                     values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr,
                       void*                values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// DENSE VECTOR DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                    int64_t               size,
                    void*                 values,
                    cudaDataType          valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnVecGet(const cusparseDnVecDescr_t dnVecDescr,
                 int64_t*                   size,
                 void**                     values,
                 cudaDataType*              valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnVecGetValues(const cusparseDnVecDescr_t dnVecDescr,
                       void**                     values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr,
                       void*                values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// SPARSE MATRIX DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 cooRowInd,
                  void*                 cooColInd,
                  void*                 cooValues,
                  cusparseIndexType_t   cooIdxType,
                  cusparseIndexBase_t   idxBase,
                  cudaDataType          valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 csrRowOffsets,
                  void*                 csrColInd,
                  void*                 csrValues,
                  cusparseIndexType_t   csrRowOffsetsType,
                  cusparseIndexType_t   csrColIndType,
                  cusparseIndexBase_t   idxBase,
                  cudaDataType          valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr,
                     int64_t               rows,
                     int64_t               cols,
                     int64_t               nnz,
                     void*                 cooInd,
                     void*                 cooValues,
                     cusparseIndexType_t   cooIdxType,
                     cusparseIndexBase_t   idxBase,
                     cudaDataType          valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCooGet(const cusparseSpMatDescr_t spMatDescr,
               int64_t*                   rows,
               int64_t*                   cols,
               int64_t*                   nnz,
               void**                     cooRowInd,  // COO row indices
               void**                     cooColInd,  // COO column indices
               void**                     cooValues,  // COO values
               cusparseIndexType_t*       idxType,
               cusparseIndexBase_t*       idxBase,
               cudaDataType*              valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCooAoSGet(const cusparseSpMatDescr_t spMatDescr,
                  int64_t*                   rows,
                  int64_t*                   cols,
                  int64_t*                   nnz,
                  void**                     cooInd,     // COO indices
                  void**                     cooValues,  // COO values
                  cusparseIndexType_t*       idxType,
                  cusparseIndexBase_t*       idxBase,
                  cudaDataType*              valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseCsrGet(const cusparseSpMatDescr_t spMatDescr,
               int64_t*                   rows,
               int64_t*                   cols,
               int64_t*                   nnz,
               void**                     csrRowOffsets,
               void**                     csrColInd,
               void**                     csrValues,
               cusparseIndexType_t*       csrRowOffsetsType,
               cusparseIndexType_t*       csrColIndType,
               cusparseIndexBase_t*       idxBase,
               cudaDataType*              valueType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetFormat(const cusparseSpMatDescr_t spMatDescr,
                       cusparseFormat_t*          format)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetIndexBase(const cusparseSpMatDescr_t spMatDescr,
                          cusparseIndexBase_t*       idxBase)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetValues(const cusparseSpMatDescr_t spMatDescr,
                       void**                     values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr,
                       void*                values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                             int                  batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetStridedBatch(const cusparseSpMatDescr_t spMatDescr,
                             int*                       batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// DENSE MATRIX DESCRIPTOR

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                    int64_t               rows,
                    int64_t               cols,
                    int64_t               ld,
                    void*                 values,
                    cudaDataType          valueType,
                    cusparseOrder_t       order)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnMatGet(const cusparseDnMatDescr_t dnMatDescr,
                 int64_t*                   rows,
                 int64_t*                   cols,
                 int64_t*                   ld,
                 void**                     values,
                 cudaDataType*              type,
                 cusparseOrder_t*           order)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetValues(const cusparseDnMatDescr_t dnMatDescr,
                       void**                     values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr,
                       void*                values)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                             int                  batchCount,
                             int64_t              batchStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetStridedBatch(const cusparseDnMatDescr_t dnMatDescr,
                             int*                       batchCount,
                             int64_t*                   batchStride)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// SPARSE VECTOR-VECTOR MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseSpVV(cusparseHandle_t           handle,
             cusparseOperation_t        opX,
             const cusparseSpVecDescr_t vecX,
             const cusparseDnVecDescr_t vecY,
             void*                      result,
             cudaDataType               computeType,
             void*                      externalBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpVV_bufferSize(cusparseHandle_t           handle,
                        cusparseOperation_t        opX,
                        const cusparseSpVecDescr_t vecX,
                        const cusparseDnVecDescr_t vecY,
                        const void*                result,
                        cudaDataType               computeType,
                        size_t*                    bufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// SPARSE MATRIX-VECTOR MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseSpMV(cusparseHandle_t           handle,
             cusparseOperation_t        opA,
             const void*                alpha,
             const cusparseSpMatDescr_t matA,
             const cusparseDnVecDescr_t vecX,
             const void*                beta,
             const cusparseDnVecDescr_t vecY,
             cudaDataType               computeType,
             cusparseSpMVAlg_t          alg,
             void*                      externalBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMV_bufferSize(cusparseHandle_t           handle,
                        cusparseOperation_t        opA,
                        const void*                alpha,
                        const cusparseSpMatDescr_t matA,
                        const cusparseDnVecDescr_t vecX,
                        const void*                beta,
                        const cusparseDnVecDescr_t vecY,
                        cudaDataType               computeType,
                        cusparseSpMVAlg_t          alg,
                        size_t*                    bufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


//------------------------------------------------------------------------------
// SPARSE MATRIX-MATRIX MULTIPLICATION

cusparseStatus_t CUSPARSEAPI
cusparseSpMM(cusparseHandle_t           handle,
             cusparseOperation_t        opA,
             cusparseOperation_t        opB,
             const void*                alpha,
             const cusparseSpMatDescr_t matA,
             const cusparseDnMatDescr_t matB,
             const void*                beta,
             cusparseDnMatDescr_t       matC,
             cudaDataType               computeType,
             cusparseSpMMAlg_t          alg,
             void*                      externalBuffer)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


cusparseStatus_t CUSPARSEAPI
cusparseSpMM_bufferSize(cusparseHandle_t           handle,
                        cusparseOperation_t        opA,
                        cusparseOperation_t        opB,
                        const void*                alpha,
                        const cusparseSpMatDescr_t matA,
                        const cusparseDnMatDescr_t matB,
                        const void*                beta,
                        cusparseDnMatDescr_t       matC,
                        cudaDataType               computeType,
                        cusparseSpMMAlg_t          alg,
                        size_t*                    bufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/******* cudart *********/
__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

// __host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }
//
// __host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }
//
// __host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }

// __host__ cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }
//
// __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr)
// {
//     fprintf(stderr, "%s is not implemented\n", __func__);
//     abort();
// }

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void);

__host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error)
{
    const char *ret = ava_execute();
    ava_return_value {
        ava_out; ava_buffer(strlen(ret) + 1);
        ava_lifetime_static;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetDeviceFlags( unsigned int *flags )
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    ava_argument(pStream) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    ava_argument(stream) ava_handle;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
    ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
{
    ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    ava_argument(event) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t extMem)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

ava_utility cudaError_t __helper_func_get_attributes(struct cudaFuncAttributes *attr,
                                                     struct fatbin_function *func,
                                                     const void *hostFun)
{
    if (func == NULL) {
        DEBUG_PRINT("func is NULL");
        return (cudaError_t) cudaErrorInvalidDeviceFunction;
    }

    if (func->hostfunc != hostFun) {
        fprintf(stderr, "search host func %p -> stored %p (device func %p)\n",
                hostFun, (void *)func->hostfunc, (void *)func->cufunc);
    }
    else {
        DEBUG_PRINT("matched host func %p -> device func %p\n", hostFun, (void *)func->cufunc);
    }

    CUresult ret;
    ret = cuFuncGetAttribute((int *)&attr->sharedSizeBytes,
                             CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func->cufunc);
    ret = cuFuncGetAttribute((int *)&attr->constSizeBytes,
                             CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func->cufunc);
    ret = cuFuncGetAttribute((int *)&attr->localSizeBytes,
                             CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func->cufunc);
    ret = cuFuncGetAttribute(&attr->maxThreadsPerBlock,
                             CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func->cufunc);
    ret = cuFuncGetAttribute(&attr->numRegs,
                             CU_FUNC_ATTRIBUTE_NUM_REGS, func->cufunc);
    ret = cuFuncGetAttribute(&attr->ptxVersion,
                             CU_FUNC_ATTRIBUTE_PTX_VERSION, func->cufunc);
    ret = cuFuncGetAttribute(&attr->binaryVersion,
                             CU_FUNC_ATTRIBUTE_BINARY_VERSION, func->cufunc);
    attr->cacheModeCA = 0;
    ret = cuFuncGetAttribute(&attr->maxDynamicSharedSizeBytes,
                             CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func->cufunc);
    ret = cuFuncGetAttribute(&attr->preferredShmemCarveout,
                             CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, func->cufunc);

    return (cudaError_t) ret;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    ava_disable_native_call;

    ava_argument(attr) {
        ava_out; ava_buffer(1);
    }
    ava_argument(func) {
        ava_opaque;
    }

    cudaError_t ret;
    if (ava_is_worker) {
        ret = __helper_func_get_attributes(attr, ava_metadata(func)->func, func);
        return ret;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

ava_utility cudaError_t
__helper_occupancy_max_active_blocks_per_multiprocessor(int *numBlocks,
                                                        struct fatbin_function *func,
                                                        const void *hostFun,
                                                        int blockSize,
                                                        size_t dynamicSMemSize)
{
    if (func == NULL) {
        DEBUG_PRINT("func is NULL");
        return (cudaError_t) cudaErrorInvalidDeviceFunction;
    }

    if (func->hostfunc != hostFun) {
        fprintf(stderr, "search host func %p -> stored %p (device func %p)\n",
                hostFun, (void *)func->hostfunc, (void *)func->cufunc);
    }
    else {
        DEBUG_PRINT("matched host func %p -> device func %p\n", hostFun, (void *)func->cufunc);
    }
    cudaError_t ret = cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,
        func->cufunc, blockSize, dynamicSMemSize);
    return ret;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                              const void *func,
                                              int blockSize,
                                              size_t dynamicSMemSize)
{
    ava_disable_native_call;

    ava_argument(numBlocks) {
        ava_out; ava_buffer(1);
    }

    ava_argument(func) {
        ava_opaque;
    }
    cudaError_t ret;
    if (ava_is_worker) {
        ret = __helper_occupancy_max_active_blocks_per_multiprocessor(numBlocks,
            ava_metadata(func)->func, func, blockSize, dynamicSMemSize);
        return ret;
    }
}

ava_utility cudaError_t
__helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(int *numBlocks,
                                                                   struct fatbin_function *func,
                                                                   const void *hostFun,
                                                                   int blockSize,
                                                                   size_t dynamicSMemSize,
                                                                   unsigned int flags)
{
    if (func == NULL) {
        DEBUG_PRINT("func is NULL");
        return (cudaError_t) cudaErrorInvalidDeviceFunction;
    }

    if (func->hostfunc != hostFun) {
        fprintf(stderr, "search host func %p -> stored %p (device func %p)\n",
                hostFun, (void *)func->hostfunc, (void *)func->cufunc);
    }
    else {
        DEBUG_PRINT("matched host func %p -> device func %p\n", hostFun, (void *)func->cufunc);
    }
    cudaError_t ret = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,
        func->cufunc, blockSize, dynamicSMemSize, flags);
    return ret;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks,
                                                       const void *func,
                                                       int blockSize,
                                                       size_t dynamicSMemSize,
                                                       unsigned int flags)
{
    ava_disable_native_call;

    ava_argument(numBlocks) {
        ava_out; ava_buffer(1);
    }

    ava_argument(func) {
        ava_opaque;
    }

    cudaError_t ret;
    if (ava_is_worker) {
        ret = __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(numBlocks,
            ava_metadata(func)->func, func, blockSize, dynamicSMemSize, flags);
        return ret;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

ava_begin_replacement;
__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size)
{
    *ptr = malloc(size);
    if (ptr)
        return cudaSuccess;
    else
        return cudaErrorMemoryAllocation;
}

__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    free(ptr);
    return cudaSuccess;
}
ava_end_replacement;

__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}


__host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
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

__host__ cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
    /* kind is always cudaMemcpyDeviceToHost */
    ava_argument(dst) {
        ava_out; ava_buffer(count);
    }
    ava_argument(symbol) ava_opaque;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const void *symbol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphDestroyNode(cudaGraphNode_t node)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

__host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

/* ONNX */

//#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const __half *alpha,  /* host or device pointer */
                                                          const __half *const Aarray[],
                                                          int lda,
                                                          const __half *const Barray[],
                                                          int ldb,
                                                          const __half *beta,   /* host or device pointer */
                                                          __half *const Carray[],
                                                          int ldc,
                                                          int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const __half *alpha,  /* host or device pointer */
                                                                 const __half *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const __half *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const __half *beta,   /* host or device pointer */
                                                                 __half *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount)
{
    fprintf(stderr, "%s is not implemented\n", __func__);
    abort();
}

//#endif

const char *CUDNNWINAPI
cudnnGetErrorString(cudnnStatus_t status)
{
    const char *ret = ava_execute();
    ava_return_value {
        ava_out; ava_buffer(strlen(ret) + 1);
        ava_lifetime_static;
    }
}
