---
title: 'Lapis: A Low-level API Specification Language'

author: 
- Arthur Peters

date: September, 2019

margin-left: 1.25in
margin-right: 1.25in
margin-top: 1.25in
margin-bottom: 1.25in
---

\clearpage{}

A Lapis specification describes semantics and properties of C APIs.
The important information falls into several categories:

1. **Data layout** which augments C data types (e.g., structs and pointers) with missing information (e.g., the size of dynamic buffers) required to unambiguously define a data structure's storage layout.
  This layout information allows a Lapis compiler to traverse data structures to serialize or copy them among other uses.
3. **Data semantics** required to recreate a semantically equivalent data layout and to determine which library data can be accessed by client code and visa versa.
  This includes, information about function pointers and special memory allocation functions.
2. **Data validity** guarantees, including when function outputs are valid and when the library can and should access client provided data.
  This information allows a Lapis compiler to determine, for example, when data needs to be copied from the library to the client and visa versa in an API remoting system like AvA.
  
In Lapis, this information is written in a C-like syntax that parallels the unmodified C declarations.
This syntax is designed to allow Lapis based tools to use an unmodified C parser.
A more aesthetically pleasing syntax would be possible if we extended the C parser.

**Document Note:**
This document is still in progress.
It is currently a mix of documentation of Lapis as implemented by AvA and Lapis as it will be in the future.

# API Annotations

```c
ava_name("name");
ava_version("version");
ava_identifier(id);
ava_number(id_number);
ava_soname(so_name);
```
Specify the name, version, and identifiers of the API as a whole.
The `id` is a short C identifier used to differential API related symbol names in Lapis compilers.
The `id_number` is an integer which used by the Lapis compiler to identify the API when a number is needed.
The `id_number` should be unique within a configuration so as distinguish all APIs used.
The `ava_soname` is the name for the generated library file (e.g. libguestlib.so). Default is "guestlib".

```c
ava_cflags(flags);
ava_libs(libs);
ava_export_qualifier(keywords);
```
Specify build configuration for the API.
This allows a Lapis compiler to build programs or other libraries using the API library or to generate correct code to use the library.
Similarly, the build information can be used to build stubs which implement the library API.  

```c
ava_worker_srcs(filenames...)
ava_guestlib_srcs(filenames...)
```
Specify worker-specific and guestlib-specific source files.

```c
#include <header.h>
```
Specify a header files (`header.h`) which declares functions in the API.
The API can have multiple header files.

```c
ava_guestlib_[init|fini]_[prologue|epilogue](utility_1; utility_2; ...)
ava_worker_init_epilogue(utility_1; utility_2; ...)
```
Specify utility functions to be called inside the guestlib's constructor and
destructor and worker's destructor. "Prologue" means the functions are called
right after the initialization of the endpointlib, or at the entor of the
destructor. "Epilogue" means the functions are called right before the exit
of the constructor, or right before the destroy of the endpointlib.

```c
ava_send_code(string)
ava_reply_code(string)
ava_worker_argument_process_code(string)
```
Specify code generators for sending commands from guestlib to API server,
replying commands from API server to guestlib, and ad-hoc processing argument
right before generating the calls of the wrapper functions. The strings must
be escape sequences in string literals and will be evaluated with `exec(..)`
in CAvA.


# Structural Syntax

The structural syntax do not provide information directly, but instead specify to which values or other APIs elements an annotation applies.
In Lapis, every scope is associated with a piece of the API: a function or a set of values (e.g., the value of an argument or the element values in an array).
The scope syntax generally matches variable scoping in C programs.
Annotation in a scope apply to the associated API piece and structural syntax can select "sub-pieces" (e.g., arguments of a function, or members of a structure). 

### Functions

```c
type function(argument_type argument, ...) {
  ... body ...
}
```
The body scope is associated with the function `f`.

The annotations in `body` will specify function call semantics and provide scopes to annotate the arguments and the return value, as needed. 

### Arguments and Return Values

```c
ava_argument(arg) {
  ... body ...
}
```
The body scope is associated with the argument `arg`.
This syntax can only appear in a function scope.

```c
ava_return_value {
  ... body ...
}
```
The body scope is associated with the return value of the enclosing function.
This syntax can only appear in a function scope.

### Array Elements

```c
ava_element {
  ... body ...
}
```
The body scope is associated with the elements of the immediately enclosing array value.

### `Struct`/`Union` Members

```C
ava_member(member) {
  ... body ...
}
```
The body scope is associated with the value of the specified `member` of the enclosing `struct`/`union` value.

**Implementation Note:** AvA uses the term "field" currently, instead of "member."
  

### Explicit Representation of Implicit Values

```c
ava_implicit_argument
```
Specify that a definition in a function is an implicit argument to the function which should be computed in the caller and passed with other arguments.

### Type Casts

```c
ava_type_cast(type)
```
Cast the current value to a new type. 
All annotations on this value (and nested values) are applied to the new type.
  
# Data Layout and Semantics

Data layout annotations specify how a value is layed out in memory.
Some annotations also provide semantic information to allow a Lapis compiler to generate code to interpret the values (such as, opening files referenced by filename within a value) 

## File

```c
ava_file;
```
The value is a string which identifies a file.

## Buffer

```c
ava_buffer(n);
```
The value is a pointer to an array of `n` elements.
The argument `n` may be an expression accessing any value in scope (e.g., other arguments to the function or this value itself). 
Annotations on the elements can be specified using `ava_element { ... }`.

A common case of this annotation is `ava_buffer(strlen(s) + 1)` for an argument `char *s` which contains an null terminated string.
The `+1` is required because `strlen` returns the number of bytes in the string *excluding* the null terminator, but the null terminator is a critical part of the data structure and must be included in the buffer size.
As another example, consider the arguments `float* data` and `size_t size` where `data` has `size` elements.
The annotation for `data` is `ava_argument(data) { ava_buffer(size) }` and `size` needs no annotations, since it is an [opaque](#opaque) value and that is the default.

```c
ava_buffer_allocator(allocator, deallocator);
```
The buffer value must be allocated using specialized `allocator` and `deallocator` functions.
This specifies the allocation requirements of the argument.
`allocator` and `deallocator` may be [utility](#utility-code-in-lapis) functions declared within the specification

## Value Lifetime

Values can have a lifetime that specifies how long they exist.
The lifetime of a buffer needs to be specified explicitly every time it is passed.
This is because AvA needs to know about lifetimes without examining a history of live buffers.
 
```c
ava_lifetime_coupled(obj);
```
The lifetime of the value is coupled to a specific value, `obj`.
Whenever `obj` is deallocated with `ava_deallocates`, this value's lifetime ends.

```c
ava_lifetime_call;
```
The lifetime of this value is this call.

```c
ava_lifetime_manual;
```
The lifetime of this value is continues until it is deallocated explicitly with a call with `ava_deallocates`.

```c
ava_lifetime_static;
```
The lifetime of this object is the same as the lifetime of the program.

```c
ava_deallocates;
```
The value is deallocated at or before this call.

```c
ava_allocates;
```
The value is a newly allocated object. 

## Opaque

```c
ava_opaque;
```
The value is an opaque sequence of bytes (with length provided by `sizeof`).
For simple types about which Lapis needs no additional information (e.g., `size_t` or `float`), this is sufficient.  

In Lapis, all values are opaque by default. 

## Handle

```c
ava_handle;
```
The value is an opaque handle, i.e., it is an identifier for something in the API or client that can be used later to access the same object.
A handle passed from the client to the API cannot be interpreted by the API or visa versa.
For instance, a handle return into client code by the API cannot be dereferenced by the client even if the handles type is a pointer.

In Lapis, pointers to incomplete types (e.g., undefined structures) are handles by default. 

## Callbacks

```c
ava_callback_decl ty callback(...) {
  ... body ...
}
```
Declare function pointer pointer type for use in other annotations.
The body is identical to the body of a normal API function and should specify the semantics of the calls to the callback (including the arguments and the return value).

A Lapis compiler may use this annotation to emit special code to handle function pointers differently than typical calls from the client into the library via an API function.

**Implementation Note:**
This will eventually be renamed to `ava_function_pointer_decl` since this can be used for function pointers that are not callbacks.
Other annotations may need changes to allow function pointer to be used in other cases, e.g., for dynamic function binding tools like `clGetFunctionAddress`.

```c
ava_callback(ty);
```
The value is a pointer to a function with type `ty`.
The function `ty` must be declared in the Lapis specification using `ava_callback_decl`.

```c
ava_userdata;
ava_userdata(callback_argument); // Future
```
The value is the "userdata" or "tag" value associated with the callback function pointer passed as `callback_argument`.
This value is used by the callback to find information about the specific registration of the callback.
`ava_userdata` is used used in function pointer type declarations (`ava_callback_decl`) to specify which argument contains the userdata value.
For function pointer types, there may be only one userdata argument.
`ava_userdata(...)` can be provided more than once if the same userdata is used for multiple callbacks.

**Note:** 
If a callback truely has more than one userdata argument, only one of the arguments needs to be annotated.
Others can be marked as opaque.
This restriction represents the assumption that the userdata parameter only needs to be destinguished to allow a Lapis compiler to hide stack specific implementation in the userdata pointer.

**Implementation Note:**
The current version does not support `callback_argument` and instead associates the userdata with the only callback argument to this function (making it impossible to support multiple callbacks passed to the same call with different userdata).
For callbacks which are passed to a separate call than their userdata, the current implementation requires a separate annotation:
```c
ava_callback_registration(decl);
```
The value is a callback with the provided callback declaration.
The call using this annotation need not have an ava_userdata argument, but the callback will not actually work until such a call is made and has an argument (probably an `ava_implicit_argument`) annotated with ava_callback.
Because of that, you should probably store the actual callback function pointer in `ava_metadata` for use in the later argument.

**Implementation Note:**
The current Lapis compiler does not support callbacks without userdata.
In the future, Lapis will support all callback (including those without userdata) via the `ava_callback` and `ava_userdata` annotations.
However, callbacks without userdata will always suffer from additional restrictions and overheads.
For instance, callbacks without userdata will generally require a small amount of simple runtime code generation the first time each client callback function is used.
This can be implemented by copying a template function and changing a pointer at a known offset within it, but even this simple code generation will not be possible if the client runtime environment is not allowed to mark writable pages as executable.


## Resource Accounting

Lapis supports resource usage information attached to API functions.
There are two types of resources:

1. _Instantaneous_ resources which are used for a very short amount of time to perform some operation (e.g., an ALU).
2. _Continuous_ resources which are used continuously over a period of time (e.g., memory).

Instantaneous resources represent the ability of an API implementation (e.g., an accelerator) to execute operations when requested.
These resources are accounted for by measuring the resources used by each function call.
In general, instantaneous resources are used to control the throughput of the API in some way; 
limiting the amount of the resource a client is allowed to use in a unit of time.
For example, floating-point computation is an instantaneous resource which is limited to a specific number of operations per second which must be shared between clients.  

**Note:** 
For some instantaneous resources (e.g., data transfer over the bus), the amount can be estimated based on the arguments to a function call (buffer size).
For other instantaneous resources (e.g., computation), the amount needs to be computed after the fact by measure how long the call or calls took to complete.
Because of this, resource usage annotations may be handled after a call completes by some Lapis compilers. 

Continuous resources represent the ability of the API implementation to assign some resource to a client for a period of time.
These resources are accounted for by tracking the resources assigned to each client.
In general, continuous resources are used to control the allocation of limited resources.
For example, device memory is a continuous resource which is limited by the available memory and needs to be allocated such that all clients can get the memory they need without starving other clients.

To enforce resource sharing requirements, a Lapis compiler will need to change how API calls are handled.
For continuous resources, the generated code may need to generate an artificial failure in response to an allocation request.
This requires that the compiler know how to fake a failure by constructing return values and/or executing specific code to change the library state.
For instantaneous resources, enforcement is usually a simple case of delaying certain calls until other clients have a chance to perform their instantaneous operations.

**Implementation Note:**
AvA does not currently support flexible failure construction.
This feature will be needed in the future to handle allocation failures in libraries which use both return values _and_ a static `errno`-like variable (e.g., CUDA).

**Implementation Note:**
AvA currently uses the terms "throughput" and "storage" instead of "instantaneous" and "continuous".

```c
ava_throughput_resource resource;
ava_storage_resource resource;
```
Declare a throughput or storage resource, respectively.

```c
ava_deallocates_resource(resource, amount);
```
The value is deallocated by this call.
This specifies the amount of a storage resource which is freed.

```c
ava_allocates_resource(resource, amount);
```
The value is a new object allocated by this call.
This specifies the amount of a storage resource which is allocated.

```c
ava_consumes_resource(resource, amount);
```
This function consumes `amount` of `resources` (an existing throughput resource).

## State-dependency

```c
ava_register_metadata(metadata_t)
```
Declare the type `metadata_t` (usually a C `struct`) as the metadata type used by the API.
This is not strictly required by Lapis itself, but is extremely useful for any Lapis compiler which needs to generate C code.

```c
metadata_t* ava_metadata(void* value)
```
A function returning a pointer to an instance of `metadata_t` which is associated with `value`.
Any call with the same `value` will return the same instance.
`ava_metadata` may be used in annotation expressions and prologue and epilogue code and is atomic (i.e., safe to use concurrently without any synchronization).

Generally `ava_metadata` is used to store metadata about API values. For example:
```c
void *lib_alloc(size_t size) {
  void *ret = alloc(size); // Execute the actual call. 
  ava_metadata(ret)->size = size; // Store size for later use.
  return ret;
}
void lib_free(void* buf) {
  ava_metadata(ret)->size; // Do something with size
}
``` 

**Implementation Note:** 
AvA executes the prologues and epilogues which will assign to metadata values in both the client and the worker.
This means that metadata is kept in sync between the two as long as the specification code executes the same in those places.

# Data Validity and Accessibility

```c
ava_sync;
```
The function executes synchronously and returns after its work is complete.
All of its output arguments and return values are valid when the function returns.
This is the default.
Synchronous calls also have the flushing behaviour described below.

```c
ava_async;
```
The function returns *immediately* and executes asynchronously.
Its output argument will be updated at some later time.

```c
ava_flush;
``` 
The function returns as with `ava_async`, but when this function is called all outstanding asynchronous calls (including this one) must be available for execution.
This applies in cases where the libraries execution model involves some form of queuing in which the library may delay dispatching asynchronous calls until a later time,
but the client needs to asynchronously force execution to start to allow parallelism.
[`clFlush`](https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/clFlush.html) is an example of these call semantics.
For comparison, [`clFinish`](https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/clFinish.html) is synchronous.

```c
ava_success(v);
```
The asynchronous function (annotated with `ava_async` or `ava_flush`) should return the value `v` to signal successful *dispatch*.

```c
ava_output;
```
The associated container value is filled by the call and contains valid data for the caller to read after the call.
May be combined with `ava_input`.

```c
ava_input;
```
The associated container value is filled by the caller and read by the callee.
May be combined with `ava_output`.


# Conditional Annotations

Sometimes a function or value has varying properties.
These can be provided using conditional annotations using `if` statements.

```c
if (predicate) {
  then_annotations
} else {
  else_annotations
}
```
Apply the annotations in `then_annotations` if `predicate` is true, and `else_annotations` otherwise.
The `predicate` may be any C expression.
The `else` clause may be omitted.
The annotation blocks here do *not* create new Lapis scopes and instead apply to the same value as annotations outside the if statement. 
In general, annotations may only be provided once in each scope, so the same annotation cannot be provided outside an if statement and inside.


```c
ava_is_in
ava_is_out
```
Build-in Lapis values which are true when entering a call, `ava_is_in`, or when returning from a call, `ava_is_out`.
This is useful for specifying that an argument has different semantics during call and return. 


# Collective Annotations

Many annotations will apply to every value of a type or all functions or similar groups.

```c
ava_functions { ... body ... }
```
The body scope applies to all functions.

```c
ava_type(ty) { ... body ... }
```
The body scope applies to all values with the type `ty`.

```c
ava_incomplete_types { ... body ... }
```
The body scope applies to all values with incomplete types.

**Implementation Note:**
This is currently called `ava_non_transferable_types` and also applies to pointers to incomplete types.
The exact best way to handle this is not yet decided.

```c
ava_defaults
```
Any collective annotation block can be annotated with `ava_defaults` to make the annotations only apply if *no explicit annotations are given for the value*.
This allows defaults to be given without causing errors if the defaults need to be overridden.
For example,
```c
ava_type(const char*) ava_defaults { ... }
```
specifies annotations for constant strings without preventing specific strings from having different annotations.


# Utility Code in Lapis

Lapis code allows expressions in many places.
To simplify the Lapis expressions, Lapis allows the specification to include utility functions and variables.

```c
ava_utility type f(...);
```
Declare a utility function for use elsewhere in the specification.
This syntax can also be used for global variables.

```c
ava_begin_utility;
... utility functions, variables, etc. ...
ava_end_utility;
```
Add utility code (which may include any C code, such as `#include`s of utility libraries).


```c
type function(...) {
  ... prologue ...
  ... body ...
  type ret = function(...); // Perform actual API call.
  ... epilogue ...
}
```
Any function in the specification can include a prologue and epilogue of normal C code before and after the annotations.
The prologue can declare variables (computed based on the function arguments) for use in annotations.
The epilogue can provide values for annotations only if those annotations are handled by the Lapis compiler _after_ the call (e.g., `ava_consumes_resource`).

The API call `type ret = function(...)` is _not_ a recursive call in Lapis and instead represents the actual call to the API function.
The Lapis declaration of the function is not a real C function definition, so this will never be recursive as it may appear.

The prologue and epilogue can also be used as an escape hatch to inject C code to be included by a Lapis compiler into its output.

**Implementation Note:**
Currently, AvA uses a special function `ava_execute()` instead of a call to the actual API function name.
This has caused a number of problems and will be changed.

```c
__CAVA__
__AVA__
```
These C preprocessor macros are declared during Lapis compilation and in both compilation and in any generated code, respectively.
These are useful for preprocessor hacks in cases when Lapis is not as expressive as required for the application. 

```c
ava_begin_replacement;
type function(...) {
  ...
}
ava_end_replacement;
```
Provide an implementation of an API function inside the specification.
This is useful for cases where a Lapis compiler can use a replacement for an API function to avoid whatever overheads the compiled code would generally have.
Multiple functions can be included in a single `replacement` block.

# API Remoting Specific Features

AvA extends Lapis with some API remoting features.

## Zero-copy Buffers

Buffers that are always allocated by API functions can be allocated in a special AvA memory region which is shared between the client application and API worker.

```c
ava_zerocopy_buffer;
```
The annotated value is a zero-copy buffer.
Zero-copy buffers do not need a size because they exist in a shared address-space and do not need to be interpreted or transferred by Lapis derived code.

```c
void *ava_zerocopy_alloc(size_t size);
void ava_zerocopy_free(void* ptr);
```
Allocate or deallocate a zero-copy buffer.
These are generally used within a replacement function, but can be called from within any AvA specification code (e.g., prologues).

```c
uintptr_t ava_zerocopy_get_physical_address(void* ptr);
```
Return the physical address of the zero-copy buffer as it would be used by a physical device in the system.
This is used to support kernel by-pass devices which interact via physical memory.

**Implementation Note:**
Currently all zero-copy memory supports physical addresses and is physically contiguous.
This is not practical.
This API will be split into a restricted "physical zero-copy memory" system and a more relaxed "virtual zero-copy memory" system.
"Physical zero-copy memory" will support allocations up to 2MiB (i.e., one huge page) and will not allow oversubscription.
"Virtual zero-copy memory" will allow large allocations (i.e., GiBs), but will not be physically contiguous or physically pinned so AvA will not provide its physical address.

## Call and Object Recording

AvA supports record-and-replay for migration and swapping.
The following annotations provide enough information about API objects to do this.

**Implementation Note:**
This extract/replace design works, but will not scale well in the presence of very large API objects.
This is because the design forces then entire state of the object to be stored into system memory before transfer.
This design will be modified to use an abstract "stream" as the target for extracted data and as the source for replacement.
The "stream" can either be implemented as using buffers (to emulate the current design) or be a thin wrapper over a socket for optimized migrations.

```c
ava_object_record;
```
The current function call is required to create or configure the annotated value.
This call will be replayed (along with all other calls recorded for the annotated value) to recreate the value.

```c
ava_object_depends_on(dependency)
```
The annotated object depends on object `dependency`, i.e., `dependency` must exist for the annotated value to be created.
This annotation is only needed in cases where the `dependency` is never explicitly passed as an argument to a function used to create or configure the annotated value.

In cases where replaying calls is not an effective way to create an object and the objects state can be read back using the API,
the specification author can provide explicit `extract` and `replace` functions that serialize and deserialize the object.
The call that constructs the object must still be annotated with `ava_object_record`.

```c
typedef void* (*ava_extract_function)(
    void *obj, size_t *length);
```
The type of extract function.
The extract function must extact the explicit state of `obj` and return it as a malloc'd buffer.
The caller takes ownership of the returned buffer. 
The length of the buffer must be written to `*length`.

```c
typedef void (*ava_replace_function)(
    void* obj, void* data, size_t length);
```
Replace (reconstruct) the explicit state of `obj` from `data` (which has length `length`).

```c
ava_object_explicit_state_functions(extract, replace);
```
Mark this object to use the provided `extract` and `replace` functions.
Objects without this annotation must be reconfigured entirely by recorded calls.

# Future Work

This section discusses how to implement a number of features currently missing from Lapis.

## Ordering

API function calls may be sensative to order, i.e., changing the order of the calls changes the semantics of those calls, or insensitive to order, i.e., reordering calls does not change their semantics.
Some APIs, like OpenCL, are mostly ordered, where as others, like TensorFlow C, are mostly unordered.
However the relationship between semantics and ordering is complex due to asynchrony, action streams, and other API features.

A synchronous call can be unordered with respect to other calls and an asynchronous call can be ordered.
For example, `clEnqueueTask` is ordered with respect to calls on the same command queue (otherwise `clEnqueueWriteBuffer` would not be guaranteed to complete before the task starts), but both `clEnqueueWriteBuffer` and `clEnqueueTask` are fully asynchronous.
However, `clFinish` on two different command queues need not be ordered.

A Lapis compiler can use ordering information to optimize call handling, e.g., by reordering calls to improve batching efficiency or even executing calls in parallel.

```c
ava_unordered;
```
The function is not ordered with respect to any other calls.

```c
ava_ordered_within(value);
```
This function is ordered with respect to any other calls "ordered within" the same `value`.
That is, the ordering of all calls "ordered within" the same `value` is semantically important, but calls ordered within disjoint sets of values the ordering is not semantically important. 
For example, `ava_ordered_within(stream)` specifies that the function is ordered with respect to other calls on the same stream (e.g., a CUDA stream).

`ava_ordered_within(NULL)` specifies that the call is ordered with respect to all other calls with the same annotation (i.e., ordered w.r.t. `NULL`); 
This is the default.

## Concurrency

Some API functions can be called concurrently (e.g., in parallel or in the middle of the execution of one another) without changing their semantics.
A common example of this is functions on different "streams."
[Unordered calls](#ordering) may or may not be concurrency safe since they may require that they are called in *some* order even if the exact order does not matter.

```c
ava_atomic;
```
The function is atomic and can be called concurrently with any other function.

```c
ava_within_monitor(value);
```
This function can be called concurrently with any function that does not share a "monitor value".
That is, this function must be called within the conceptual monitor of `value`, but calls with disjoint sets of monitors may execute concurrently.

`ava_within_monitor(NULL)` specifies that the call cannot execute concurrently with any other calls with the same annotation (i.e., with the monitor `NULL`); 
This is the default.

**Open Design Question:** 
There may need to be a way to specify calls cannot be concurrent *unless* they are on different threads in the client code.
This would be needed for cases where the true concurrency limitations are not easily encoded in the specification, so the specification author wishes to partially trust the client code.
In addition, some compilers (e.g., an RPC compiler) may be faster if client code threads are trusted.
It is not yet clear the best way to handle these issues.
