#include <stdlib.h>
#include <assert.h>

//////// Internal Utilities
#define __CONCATENATE_DETAIL(x, y) x##y
#define __CONCATENATE(x, y) __CONCATENATE_DETAIL(x, y)
#define __STRINGIFY_DETAIL(x) #x
#define __STRINGIFY(x) __STRINGIFY_DETAIL(x)

#define __MAKE_UNIQUE(x) __CONCATENATE(x, __CONCATENATE(_, __COUNTER__))
#define __AVA_NAME(x) __CONCATENATE(__AVA_PREFIX, x)

#define __AVA_ANNOTATE(s) __attribute__((annotate(__STRINGIFY(__AVA_PREFIX) s)))
#define __AVA_ANNOTATE_STMT_TYPED(type, name, value) ({ type __AVA_NAME(name) = value; })
#define __AVA_ANNOTATE_STMT(name, value) __AVA_ANNOTATE_STMT_TYPED(typeof(value), name, value)
#define __AVA_ANNOTATE_FLAG(name) __AVA_ANNOTATE_STMT_TYPED(int, name, 1)


struct ava_throughput_resource_t;
typedef struct ava_throughput_resource_t* ava_throughput_resource;
struct ava_storage_resource_t;
typedef struct ava_storage_resource_t* ava_storage_resource;

#define __AVA__ 1
#define __CAVA__ 1

//////// Annotations

///// Value annotations (informally, additional type information)

/// Apply annotations to the elements of the argument. This can be
/// nested to apply to elements of elements.
#define ava_element if(__AVA_NAME(element_block))
int __AVA_NAME(element_block) = 1;

/// Apply annotations to a specific field of the current value.
#define ava_field(name) int __AVA_NAME(__CONCATENATE(field_block_, name)) = 1; \
                        if(__AVA_NAME(__CONCATENATE(field_block_, name)))

/// Apply annotations to a specific argument of a function.
#define ava_argument(arg) if(__AVA_NAME(argument_block) && arg)
int __AVA_NAME(argument_block) = 1;

/// Apply annotations to the return value of a function.
#define ava_return_value if(__AVA_NAME(return_value_block))
int __AVA_NAME(return_value_block) = 1;


extern unsigned long int ava_index;

/// Provide a return value which specifies success for a given type.
#define ava_success(v) __AVA_ANNOTATE_STMT(success, v)

// TODO: failure? There could be multiple kinds of failure. It's not clear how to represent this. When will failure need to be forged?
// I think failure may need to be handled by code in the bodies of functions. Simply by returning the appropriate value.

enum ava_transfer_t {
    NW_NONE=0,
    NW_HANDLE,
    NW_OPAQUE,
    NW_BUFFER,
    NW_CALLBACK,
    NW_CALLBACK_REGISTRATION,
    NW_FILE,
    NW_ZEROCOPY_BUFFER
};

enum ava_lifetime_t {
    AVA_CALL=0,
    AVA_COUPLED,
    AVA_STATIC,
    AVA_MANUAL
};

typedef void *(*ava_buffer_allocator)(size_t size);
typedef void (*ava_buffer_deallocator)(void *ptr);

/// Specify special allocation and deallocation (free) functions to be used on the worker. The functions may be API
/// functions or utility functions (see ava_utility).
#define ava_buffer_allocator(allocator, deallocator)  ({ \
        __AVA_ANNOTATE_STMT_TYPED(ava_buffer_allocator, buffer_allocator, allocator); \
        __AVA_ANNOTATE_STMT_TYPED(ava_buffer_deallocator, buffer_deallocator, deallocator); })

/// The value is a string which identifies a file. The file will be
/// copied to the target and the filename replaced with the actual
/// filename. The filename will be interpreted relative to the current
/// working directory in the guest.
#define ava_file __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_FILE)

/// The argument is a worker side pointer. Do not copy the pointer
/// value (or nested pointer type if it exists) and instead treat the
/// value as an opaque handle. Values transferred from the invoker to
/// the client will be obfiscated and verified for security.
///
/// TODO: Currently ava_handle implicitly treats non-handles as
/// ava_opaque and this will probably change in the future (e.g., to
/// have the spec have an explicit conditional for handle and non-handle
/// cases).
#define ava_handle __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_HANDLE)

/// Treat this value as an opaque value (effectively as a intptr_t if
/// it is a pointer).
#define ava_opaque __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_OPAQUE)

/// Specify the size of this buffer *in original elements*. An original element
/// is an element of the original type provided in the function prototype. Type
/// casts do not affect the buffer size, so you must specify all the buffer sizes
/// using `sizeof(T)` if the original argument is `void*` but has been cast to `T*`.
#define ava_buffer(len) ({                          \
            __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_BUFFER); \
            __AVA_ANNOTATE_STMT_TYPED(unsigned long int, buffer, len); \
        })

/// Specify that the value is a pointer which was allocated with ava_zerocopy_alloc.
/// The pointer is adjusted to match the virtual address space in the destination, but is not otherwise changed.
#define ava_zerocopy_buffer ({ \
    __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_ZEROCOPY_BUFFER); \
    ava_lifetime_call; \
    __AVA_ANNOTATE_STMT_TYPED(unsigned long int, buffer, 4294967295); \
    ava_in; ava_out; \
    })
// The size is set to a large value since it does have a size, but that size is unknown (and not needed).

#ifdef __cplusplus
#define __FUNCTION_PTR(x) &(x)
#else
#define __FUNCTION_PTR(x) x
#endif

/// Mark an argument as a callback with the provided callback declaration. The same call
/// must have another argument annotated with ava_userdata.
#define ava_callback(decl) ({             \
            __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_CALLBACK); \
            __AVA_ANNOTATE_STMT_TYPED(void*, callback_stub_function, (void*)decl); \
        })

/// Mark an argument as a callback with the provided callback declaration. The
/// call using this annotation need not have an ava_userdata argument, but
/// The callback will not actually work until such a call is made and has an
/// argument (probably an ava_implicit_argument) annotated with ava_callback.
/// Because of that you should probably store the actual callback function
/// pointer in ava_metadata for use in the later argument.
#define ava_callback_registration(decl) ({             \
            __AVA_ANNOTATE_STMT_TYPED(enum ava_transfer_t, transfer, NW_CALLBACK_REGISTRATION); \
            __AVA_ANNOTATE_STMT_TYPED(void*, callback_stub_function, decl); \
        })


/// Specify the lifetime of the annotated value's shadow as coupled a specific value, `obj`.
/// Whenever `obj` is transported with the ava_deallocates annotation the shadow will be
/// freed.
///
/// The lifetime of a buffer needs to be specified every time it is passed. This is because
/// AvA needs to know about lifetimes without examining a history of live buffers.
#define ava_lifetime_coupled(obj) ({             \
            __AVA_ANNOTATE_STMT_TYPED(enum ava_lifetime_t, lifetime, AVA_COUPLED); \
            __AVA_ANNOTATE_STMT_TYPED(void*, lifetime_coupled, obj); \
        })

/// Specify that the value's shadow has the lifetime of this specific call.
///
/// The lifetime of a buffer needs to be specified every time it is passed. This is because
/// AvA needs to know about lifetimes without examining a history of live buffers.
#define ava_lifetime_call __AVA_ANNOTATE_STMT_TYPED(enum ava_lifetime_t, lifetime, AVA_CALL);

/// Specify that the value's shadow should live until an explicit call with ava_deallocates is made on this buffer.
///
/// The lifetime of a buffer needs to be specified every time it is passed. This is because
/// AvA needs to know about lifetimes without examining a history of live buffers.
#define ava_lifetime_manual __AVA_ANNOTATE_STMT_TYPED(enum ava_lifetime_t, lifetime, AVA_MANUAL);

/// Specify that the value's shadow has the lifetime of the whole program.
///
/// The lifetime of a buffer needs to be specified every time it is passed. This is because
/// AvA needs to know about lifetimes without examining a history of live buffers.
#define ava_lifetime_static __AVA_ANNOTATE_STMT_TYPED(enum ava_lifetime_t, lifetime, AVA_STATIC);

/* TODO: Handle reference counting
/// The handle this is attached to is reference counted, so NightWatch
/// should perform parallel reference counting for related options.
#define ava_reference_counted __AVA_ANNOTATE_FLAG("reference_counted")
*/

/// The value should be copied out of the call making it up to date in
/// the guest after the call. This annotation goes on the container
/// which should be copied out. It should also have a transfer
/// annotation.
#define ava_output __AVA_ANNOTATE_FLAG(output)
#define ava_out ava_output

/// The value should be copied in to the call making it up to date in
/// the worker at entry to the call. This annotation goes on the
/// container which should be copied in. It should also have a
/// transfer annotation.
#define ava_input __AVA_ANNOTATE_FLAG(input)
#define ava_in ava_input

/// The value should be copied in to the call making it up to date in
/// the worker at entry to the call. This annotation goes on the
/// container which should be copied in. It should also have a
/// transfer annotation.
#define ava_no_copy __AVA_ANNOTATE_FLAG(no_copy)

/// The value is deallocated by this call.
#define ava_deallocates __AVA_ANNOTATE_FLAG(deallocates)

/// The value is deallocated by this call.
/// This specifies the amount of a storage resource which is freed.
/// This annotation implies `ava_deallocates`.
#define ava_deallocates_resource(resource, amount) ({ \
    __AVA_ANNOTATE_FLAG(deallocates); \
    __AVA_ANNOTATE_STMT_TYPED(long int, __CONCATENATE(deallocates_amount_, resource), amount); })

/// The value is a new object allocated by this call. This only makes
/// sense on output arguments (return value and `out` values).
#define ava_allocates __AVA_ANNOTATE_FLAG(allocates)

/// The value is a new object allocated by this call.
/// This specifies the amount of a storage resource which is allocated.
/// This annotation implies `ava_allocates`.
#define ava_allocates_resource(resource, amount) ({ \
    __AVA_ANNOTATE_FLAG(allocates); \
    __AVA_ANNOTATE_STMT_TYPED(long int, __CONCATENATE(allocates_amount_, resource), amount); })

/////// Argument annotations

/// Mark an argument (either to a callback or to register a callback)
/// as accepting or providing user data as pointer. The API should not
/// interpret this data in any way.
#define ava_userdata __AVA_ANNOTATE_FLAG(userdata)

/// Specify that a definition in a function is an implicit argument to
/// the function which should be computed in the caller and passed
/// with other arguments.
#define ava_implicit_argument __AVA_ANNOTATE("implicit_argument")

/// Specify that an annotations on this argument depend on another
/// argument, meaning the other argument must be transferred first.
/// For instance, `depends_on(size)` will ensure that `buffer(*size)`.
///
/// Currently there is no way to make parts of an argument depend on
/// another, which makes cyclic dependancies between arguments
/// unsupported.
#define ava_depends_on(...) __AVA_ANNOTATE_STMT_TYPED(const char*, depends_on, #__VA_ARGS__)

/// Cast the current value to a new type. All annotations on this
/// value (and nested values) are applied to the new type.
#define ava_type_cast(type) ({ type __AVA_NAME(type_cast); })

/////// Functions

/// Specify that a function is a callback function type in this API.
/// It will be handled specially to support callbacks, including
/// supporting receiving this call in the guestlib.
#define ava_callback_decl __AVA_ANNOTATE("callback_decl")

#define ava_synchrony(expr) __AVA_ANNOTATE_STMT_TYPED(enum ava_sync_mode_t, synchrony, expr)

/// This function must be executed synchronously.
#define ava_sync ava_synchrony(NW_SYNC)

/// This function may be executed asynchronously. The function must
/// return void and must not have any out arguments.
#define ava_async ava_synchrony(NW_ASYNC)

/// This function may be executed asynchronously, but must cause the
/// commands to be sent to the invoker before returning. The function
/// must return void and must not have any out arguments.
#define ava_flush ava_synchrony(NW_FLUSH)

/// This function consumes `amount` of `resources` (an existing throughput resource).
/// `amount` must be convertible to `long int`.
#define ava_consumes_resource(resource, amount) ({ \
    __AVA_ANNOTATE_STMT_TYPED(long int, __CONCATENATE(consumes_amount_, resource), amount); })

// This is a utility annotation used to generate code to measure time spent in different parts
// of the offloading.
#define ava_time_me __AVA_ANNOTATE_FLAG(generate_timing_code)

#define ava_unsupported __AVA_ANNOTATE_FLAG(unsupported)

// This function's native API will not be called in the host worker.
#define ava_disable_native_call __AVA_ANNOTATE_FLAG(disable_native)

//////// Record and Replay

/// Extract the explicit state of the object `o` and return it as a malloc'd buffer.
/// The caller takes ownership of the buffer. The length of the buffer must be written to `*length`.
typedef void* (*ava_extract_function)(void *obj, size_t *length);

/// Replace (reconstruct) the explicit state of the object `o` from data (which has length `length`).
typedef void (*ava_replace_function)(void* obj, void* data, size_t length);

/// Add a dependency on object `dependency` to object `dependent`, causing `dependency` to be captured along with `dependent`.
#define ava_object_depends_on(dependency) \
    __AVA_ANNOTATE_STMT_TYPED(const void*, object_depends_on, dependency)

/// Record this call for the object `obj`. This will implicitly add dependencies on all other object handles passed
/// to the call.
#define ava_object_record __AVA_ANNOTATE_FLAG(object_record)

/// Mark this object to use the provided `extract` and `replace` functions.
#define ava_object_explicit_state_functions(extract, replace) ({ \
        __AVA_ANNOTATE_STMT_TYPED(ava_extract_function, object_explicit_state_extract, extract); \
        __AVA_ANNOTATE_STMT_TYPED(ava_replace_function, object_explicit_state_replace, replace); })


//////// Category annotation declarations

/// Apply annotations to type `ty`.
#define ava_type(ty) ty* __MAKE_UNIQUE(__AVA_NAME(category_type_))()

/// Apply annotations to all functions.
#define ava_functions void __MAKE_UNIQUE(__AVA_NAME(category_functions_))()

/// Apply annotations to all pointer types.
#define ava_pointer_types void __MAKE_UNIQUE(__AVA_NAME(category_pointer_types_))()
/// Apply annotations to all const pointer types.
#define ava_const_pointer_types void __MAKE_UNIQUE(__AVA_NAME(category_const_pointer_types_))()
/// Apply annotations to all non-const pointer types.
#define ava_nonconst_pointer_types void __MAKE_UNIQUE(__AVA_NAME(category_nonconst_pointer_types_))()

/// Apply annotations to all types that point to non-portable data
/// (e.g., pointers or FILE handles). This will also apply if the
/// non-portable data is indirectly referenced by a pointer to a
/// pointer (or similar).
#define ava_non_transferable_types void __MAKE_UNIQUE(__AVA_NAME(category_non_transferable_types_))()

/// When applied to a category, the annotations applied in this
/// statement only apply when only other default annotations are
/// available for a given annotable entity.
#define ava_defaults __AVA_ANNOTATE("default")


//////// Global declarations

/// The name of the API as a string. This is only used for
/// documentation and error reporting.
#define ava_name(n) const char* __AVA_NAME(name) = n

/// The version of the API as a string. This is used for
/// documentation, error reporting, and passed to version checking
/// functions.
#define ava_version(n) const char* __AVA_NAME(version) = n

/// The internal identifier of this API. This must be a valid
/// identifier the target language and in C.
#define ava_identifier(n) const char* __AVA_NAME(identifier) = __STRINGIFY(n)

/// The internal ID number of this API. This must be unique across all
/// APIs used in the same hypervisor.
#define ava_number(n) const char* __AVA_NAME(number) = __STRINGIFY(n)

/// A qualifier string to apply to each exported function in the
/// generated API implementation. For example, for Windows supporting
/// libraries this will often be "dllexport".
#define ava_export_qualifier(n) const char* __AVA_NAME(export_qualifier) = __STRINGIFY(n)

#define ava_cflags(n) const char* __AVA_NAME(cflags) = __STRINGIFY(n)
#define ava_cxxflags(n) const char* __AVA_NAME(cxxflags) = __STRINGIFY(n)
#define ava_libs(n) const char* __AVA_NAME(libs) = __STRINGIFY(n)

/// Mark a function or variable as a utility for the rest of the specification.
/// These definitions will be passed through to the generated code without any changes.
/// They are always static (in the C sense of static linkage).
#define ava_utility static

/// Begin a region of utility code that will be passed through to the generated
/// code without being changed or interpreted.
#define ava_begin_utility int __MAKE_UNIQUE(__AVA_NAME(begin_utility)) = 1

/// End a region of utility code.
/// See ava_begin_utility.
#define ava_end_utility int __MAKE_UNIQUE(__AVA_NAME(end_utility)) = 0

/// Begin a region of replacement code that will be provided to the application
/// as part the guestlib without being changed or interpreted. This is used
/// to replace library functions with custom paravirtual versions.
#define ava_begin_replacement int __MAKE_UNIQUE(__AVA_NAME(begin_replacement)) = 1

/// End a region of replacement code.
/// See ava_begin_replacement.
#define ava_end_replacement int __MAKE_UNIQUE(__AVA_NAME(end_replacement)) = 0

/////// Enums

enum ava_sync_mode_t {
  NW_ASYNC = 0,
  NW_SYNC,
  NW_FLUSH
};

/// True iff the expression is executing in the worker.
extern int ava_is_worker;
/// True iff the expression is executing in the guest.
extern int ava_is_guest;

/// True iff the expression is executing during function in argument processing.
extern int ava_is_in;
/// True iff the expression is executing during function out argument processing.
extern int ava_is_out;

/// Register `metadata_t` as the metadata type for this API. This must be provided to enable object metadata.
#define ava_register_metadata(metadata_t) metadata_t* ava_metadata(const void* const)

/////// Defaults

ava_non_transferable_types {
    ava_handle;
}

ava_const_pointer_types ava_defaults {
    ava_in;
    ava_buffer(1);
}
ava_nonconst_pointer_types ava_defaults {
    ava_in;
    ava_out;
    ava_buffer(1);
}

/////// Utility functions

unsigned long int max(unsigned long int a, unsigned long int b);
unsigned long int min(unsigned long int a, unsigned long int b);
int DEBUG_PRINT(const char *fmt, ...);

#include <stdint.h>
#include <ctype.h>
#include <string.h>

void *ava_zerocopy_alloc(size_t size);
void ava_zerocopy_free(void* ptr);
uintptr_t ava_zerocopy_get_physical_address(void* ptr);


struct __ava_unknown;

/// Used within function to trigger the invocation of the underlying
/// API. The actual return type is the return type of the API call. If
/// the return value is captured, the variable name must be "ret".
struct __ava_unknown* ava_execute();
