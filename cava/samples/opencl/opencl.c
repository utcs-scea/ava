// clang-format off
ava_name("OpenCL");
ava_version("1.2");
ava_identifier(OpenCL);
ava_number(2);
ava_soname(libOpenCL.so libOpenCL.so.1);
//ava_cflags(-DAVA_PRINT_TIMESTAMP);
//ava_cflags(-DAVA_RECORD_REPLAY);
//ava_cflags(-DAVA_RECORD_REPLAY -DAVA_BENCHMARKING_MIGRATE);
//ava_cflags(-DAVA_API_FUNCTION_CALL_RESOURCE -DAVA_DISABLE_HANDLE_TRANSLATION);
ava_libs(-lOpenCL);
ava_export_qualifier(CL_API_ENTRY);
// clang-format on

ava_functions ava_defaults { ava_sync; }

ava_functions { ava_time_me; }

ava_non_transferable_types { ava_handle; }

#include <CL/cl.h>

ava_begin_utility;
#include "common/logging.h"
ava_end_utility;

ava_type(cl_int) { ava_success(CL_SUCCESS); }

typedef struct {
  size_t size;
  void *related_buffer;
  cl_bitfield flags;

  /* buffer object */
  size_t buffer_size;
  cl_context context;
  cl_device_id devices[10];

  /* argument types */
  int kernel_argc;
  char kernel_arg_is_handle[64];
} Metadata;

ava_register_metadata(Metadata);

ava_throughput_resource command_rate;
ava_storage_resource device_memory;

ava_callback_decl void cl_pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  ava_argument(errinfo) {
    ava_buffer(strlen(errinfo) + 1);
    ava_in;
  }
  ava_argument(private_info) {
    ava_buffer(cb);
    ava_in;
  }
  ava_argument(user_data) { ava_userdata; }
}

#ifndef ava_min
#define ava_min
#define ava_min(a, b) (a < b ? a : b)
#endif
#define cl_in_out_buffer(x, y)                    \
  ({                                              \
    if (ava_is_in)                                \
      ava_buffer(x);                              \
    else                                          \
      ava_buffer(ava_min(x, y == NULL ? x : *y)); \
  })

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  ava_argument(platforms) {
    ava_out;
    cl_in_out_buffer(num_entries, num_platforms);
    ava_element ava_object_record;
  }
  ava_argument(num_platforms) {
    ava_out;
    ava_buffer(1);
  }
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
}

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
  ava_argument(strings) {
    ava_in;
    ava_buffer(count);
    ava_element ava_buffer((lengths != NULL && lengths[ava_index]) ? lengths[ava_index]
                                                                   : strlen(strings[ava_index]) + 1);
  }
  ava_argument(lengths) {
    ava_in;
    ava_buffer(count);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates;
    ava_object_record;
    ava_object_depends_on(context);
  }
}

cl_int clReleaseProgram(cl_program program) {
  ava_async;
  ava_argument(program) {
    ava_deallocates;
    ava_object_record;
  }
}

ava_utility size_t clEnqueueRWImage_ptr_size(size_t row_pitch, size_t slice_pitch, const size_t *region) {
#warning image size is not computed
  return 0;
}

cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t *origin,
                           const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr,
                           cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  if (blocking_write == CL_TRUE)
    ava_sync;
  else
    ava_async;
  ava_argument(origin) {
    ava_in;
    ava_buffer(3);
  }
  ava_argument(region) {
    ava_in;
    ava_buffer(3);
  }
  ava_argument(ptr) {
    ava_in;
    ava_buffer(clEnqueueRWImage_ptr_size(input_row_pitch, input_slice_pitch, region));
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element ava_allocates;
  }
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
  ava_argument(devices) {
    ava_out;
    cl_in_out_buffer(num_entries, num_devices);
    ava_element {
      ava_object_depends_on(platform);
      ava_object_record;
    }
  }
  ava_argument(num_devices) {
    ava_out;
    ava_buffer(1);
  }
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
}

ava_utility size_t clCreateContext_properties_size(const cl_context_properties *properties) {
  size_t size = 1;
  while (properties[size - 1] != 0) {
    size++;
  }
  return size;
}

cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
                           void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                           cl_int *errcode_ret) {
  ava_object_record;
  ava_argument(properties) {
    ava_in;
    ava_buffer(clCreateContext_properties_size(properties));
    ava_element {
      if ((ava_index > 0) && (ava_index & 1) && properties[ava_index - 1] == CL_CONTEXT_PLATFORM) {
        ava_handle;
      }
    }
  }

  ava_argument(devices) {
    ava_in;
    ava_buffer(num_devices);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(pfn_notify) { ava_callback(cl_pfn_notify); }
  ava_argument(user_data) { ava_userdata; }
  ava_return_value {
    ava_allocates;
    ava_object_record;
  }
}

cl_context clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                                   cl_int *errcode_ret) {
  ava_object_record;
  ava_argument(properties) {
    ava_in;
    ava_buffer(clCreateContext_properties_size(properties));
    ava_element {
      if ((ava_index > 0) && (ava_index & 1) && properties[ava_index - 1] == CL_CONTEXT_PLATFORM) {
        ava_handle;
      }
    }
  }
  ava_argument(pfn_notify) { ava_callback(cl_pfn_notify); }
  ava_argument(user_data) { ava_userdata; }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates;
    ava_object_record;
  }
}

cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
    if (ava_is_worker && param_name == CL_CONTEXT_DEVICES) {
      ava_type_cast(void **);
      ava_element {
        ava_handle;
        ava_object_depends_on(context);
        ava_object_record;
      }
    } else {
      ava_element ava_opaque;
    }
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
}

cl_int clReleaseContext(cl_context context) {
  ava_async;
  ava_argument(context) {
    ava_deallocates;
    ava_object_record;
  }
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates;
    ava_object_depends_on(context);
    ava_object_depends_on(device);
    ava_object_record;
  }

  cl_command_queue ret = ava_execute();
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  ava_async;
  ava_argument(command_queue) {
    ava_deallocates;
    ava_object_record;
  }
}

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  ava_argument(kernel_name) {
    ava_in;
    ava_buffer(strlen(kernel_name) + 1);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates;
    ava_object_record;
    ava_object_depends_on(program);
  }

  cl_kernel ret = ava_execute();
  char *arg_type;
  cl_uint arg_num;
  cl_uint arg_addr;
  int i;
  if (ava_is_worker) {
    arg_type = alloca(32);
    clGetKernelInfo(ret, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &arg_num, NULL);
    ava_metadata(ret)->kernel_argc = arg_num;

    for (i = 0; i < arg_num; i++) {
      if (clGetKernelArgInfo(ret, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_type), arg_type, NULL) < 0) {
        clGetKernelArgInfo(ret, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(arg_addr), &arg_addr, NULL);
        ava_debug("arg#%d address qualifier=%X\n", i, arg_addr);
        if (arg_addr == CL_KERNEL_ARG_ADDRESS_PRIVATE)
          ava_metadata(ret)->kernel_arg_is_handle[i] = 0;
        else
          ava_metadata(ret)->kernel_arg_is_handle[i] = 1;
      } else {
        ava_metadata(ret)->kernel_arg_is_handle[i] =
            ((arg_type && *arg_type && arg_type[strlen(arg_type) - 1] == '*') ? 1 : 0);
        ava_debug("arg#%d has type=%s\n", i, arg_type);
      }
    }
  }
}

cl_int clReleaseKernel(cl_kernel kernel) {
  ava_argument(kernel) {
    ava_deallocates;
    ava_object_record;
  }
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void (*pfn_notify)(cl_program, void *), void *user_data) {
  ava_argument(program) ava_object_record;
  ava_argument(device_list) {
    ava_in;
    ava_buffer(num_devices);
#warning depends on buffer elements
  }
  ava_argument(options) {
    ava_in;
    ava_buffer(strlen(options) + 1);
  }
  ava_argument(pfn_notify) { ava_callback(cl_pfn_notify); }
  ava_argument(user_data) { ava_userdata; }

  char *new_options;
  if (ava_is_worker) {
    new_options = (char *)malloc((options ? strlen(options) : 0) + 21);
    if (options)
      strcpy(new_options, options);
    else
      new_options[0] = '\0';
    options = new_options;
    strcat(new_options, " -cl-kernel-arg-info");
  }
  ava_execute();
  if (ava_is_worker) {
    free(options);
  }
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
}

ava_utility void *cl_mem_extract(void *obj, size_t *length) {
  void *buf;
  printf("cl_mem_extract: obj=%lx, context=%lx\n", (uintptr_t)obj, (uintptr_t)ava_metadata(obj)->context);

  clGetContextInfo(ava_metadata(obj)->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * 10,
                   ava_metadata(obj)->devices, NULL);

  cl_command_queue cmd_q = clCreateCommandQueue(ava_metadata(obj)->context, ava_metadata(obj)->devices[0], NULL, NULL);

  *length = ava_metadata(obj)->buffer_size;
  buf = malloc(*length);
  clEnqueueReadBuffer(cmd_q, obj, CL_TRUE, 0, *length, buf, 0, NULL, NULL);
  clReleaseCommandQueue(cmd_q);

  return buf;
}

ava_utility void cl_mem_replace(void *obj, void *data, size_t length) {
  printf("cl_mem_replace: obj=%lx, ctx=%lx\n", (uintptr_t)obj, (uintptr_t)ava_metadata(obj)->context);

  clGetContextInfo(ava_metadata(obj)->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * 10,
                   ava_metadata(obj)->devices, NULL);

  cl_command_queue cmd_q = clCreateCommandQueue(ava_metadata(obj)->context, ava_metadata(obj)->devices[0], NULL, NULL);

  clEnqueueWriteBuffer(cmd_q, obj, CL_TRUE, 0, length, data, 0, NULL, NULL);
  clReleaseCommandQueue(cmd_q);
}

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
  ava_argument(host_ptr) {
    ava_in;
    // if(flags & CL_MEM_USE_HOST_PTR) buffer_lifetime(ret)
    ava_buffer(size);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates_resource(device_memory, size);
    ava_object_explicit_state_functions(cl_mem_extract, cl_mem_replace);
    ava_object_record;
    ava_object_depends_on(context);
  }

  cl_mem ret = ava_execute();
  ava_metadata(ret)->buffer_size = size;
  ava_metadata(ret)->context = context;
}

cl_int clReleaseMemObject(cl_mem memobj) {
  ava_async;
  ava_argument(memobj) {
    ava_deallocates_resource(device_memory, ava_metadata(memobj)->buffer_size);
    ava_object_record;
  }
}

ava_utility size_t clCreateImage_host_ptr_size(const cl_image_desc *image_format) {
  switch (image_format->image_type) {
  case CL_MEM_OBJECT_IMAGE2D:
    return image_format->image_row_pitch * image_format->image_height;
  case CL_MEM_OBJECT_IMAGE3D:
    return image_format->image_slice_pitch * image_format->image_depth;
  case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    return image_format->image_slice_pitch * image_format->image_array_size;
  case CL_MEM_OBJECT_IMAGE1D:
    return image_format->image_row_pitch;
  case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    return image_format->image_row_pitch * image_format->image_array_size;
  case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    return image_format->image_row_pitch;
  }
  abort();
}

ava_utility size_t clCreateImage_dims(const cl_image_desc *image_format) {
  switch (image_format->image_type) {
  case CL_MEM_OBJECT_IMAGE1D:
  case CL_MEM_OBJECT_IMAGE1D_ARRAY:
  case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    return 1;
  case CL_MEM_OBJECT_IMAGE2D_ARRAY:
  case CL_MEM_OBJECT_IMAGE2D:
    return 2;
  case CL_MEM_OBJECT_IMAGE3D:
    return 3;
  }
  abort();
}

ava_utility size_t clCreateImage2D_host_ptr_size(size_t image_height, size_t image_row_pitch) {
  return image_height * image_row_pitch;
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
                     const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
  ava_argument(image_format) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(image_desc) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(host_ptr) {
    ava_in;
    // if(flags & CL_MEM_USE_HOST_PTR) buffer_lifetime(ret)
    ava_buffer(clCreateImage_host_ptr_size(image_desc));
  }
  ava_argument(errcode_ret) { ava_out, ava_buffer(1); }
  ava_return_value ava_allocates;
}

cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t image_width,
                       size_t image_height, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
  ava_argument(image_format) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(host_ptr) {
    ava_in;
    // if(flags & CL_MEM_USE_HOST_PTR) buffer_lifetime(ret)
    ava_buffer(clCreateImage2D_host_ptr_size(image_height, image_row_pitch));
  }
  ava_argument(errcode_ret) { ava_out, ava_buffer(1); }
  ava_return_value ava_allocates;
}

cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
                                  size_t src_offset, const size_t *dst_origin, const size_t *region,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  ava_argument(dst_origin) {
    ava_in;
    ava_buffer(3);
  }
  ava_argument(region) {
    ava_in;
    ava_buffer(3);
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
}

cl_int clGetSupportedImageFormats(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type,
                                  cl_uint num_entries, cl_image_format *image_formats, cl_uint *num_image_formats) {
  ava_argument(image_formats) {
    ava_out;
    ava_buffer(num_entries);
  }
  ava_argument(num_image_formats) {
    ava_out;
    ava_buffer(1);
  }
}

ava_utility void **clWaitForEvents_event_list_to_buffer_list(cl_uint num_events, const cl_event *event_list) {
  void **buffer_list = (void **)malloc(num_events * sizeof(void *));
  for (int i = 0; i < num_events; i++) {
    buffer_list[i] = ava_metadata(event_list[i])->related_buffer;
  }
  return buffer_list;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  ava_implicit_argument void **related_buffers = clWaitForEvents_event_list_to_buffer_list(num_events, event_list);
  ava_argument(related_buffers) {
    ava_out;
    ava_depends_on(num_events, event_list);
    ava_buffer(num_events);
    ava_element ava_buffer(ava_metadata(event_list[ava_index])->size);
    ava_deallocates;
  }

  ava_argument(event_list) {
    ava_in;
    ava_buffer(num_events);
  }

  ava_execute();
#warning fix me
}

cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  ava_implicit_argument void *output_buffer = ava_metadata(event)->related_buffer;
  ava_argument(output_buffer) {
    if (param_name == CL_EVENT_COMMAND_EXECUTION_STATUS && *((cl_int *)param_value) == CL_COMPLETE) ava_out;
    ava_buffer(ava_metadata(event)->size);
  }

  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_execute();
#warning fix me
}

cl_int clReleaseEvent(cl_event event) { ava_argument(event) ava_deallocates; }

cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
}

ava_callback_decl void clSetEventCallback_callback(cl_event event, cl_int status, void *user_data) {
  ava_implicit_argument void *output_buffer = ava_metadata(event)->related_buffer;

  ava_argument(output_buffer) {
    ava_in;
    ava_buffer(ava_metadata(event)->size);
  }

  ava_argument(user_data) { ava_userdata; }

  ava_execute();
  if (ava_is_guest) clReleaseEvent(event);
}

cl_int clSetEventCallback(cl_event event, cl_int command_exec_callback_type,
                          void (*pfn_notify)(cl_event, cl_int, void *), void *user_data) {
  ava_argument(pfn_notify) { ava_callback(clSetEventCallback_callback); }
  ava_argument(user_data) { ava_userdata; }
  if (ava_is_worker) clRetainEvent(event);
  ava_execute();
}

#if DISABLED
void *clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem ava_buffer, cl_bool blocking_map,
                         cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret) {
  void *ret = ava_execute();
  ava_metadata(ret)->size = size;
  ava_metadata(ret)->flags = map_flags;
  ava_metadata(*event)->related_buffer = ret;

  if (blocking_map == CL_TRUE)
    ava_sync;
  else
    ava_async;

  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates;
    if (map_flags != CL_MAP_WRITE_INVALIDATE_REGION) {
      ava_out;
      ava_buffer(size);
    }
  }
#warning fix me
}

void *clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags,
                        const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
                        cl_int *errcode_ret) {
  if (blocking_map == CL_TRUE)
    ava_sync;
  else
    ava_async;

  ava_argument(origin) {
    ava_in;
    ava_buffer(3);
  }
  ava_argument(region) {
    ava_in;
    ava_buffer(3);
  }
  ava_argument(image_row_pitch) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(image_slice_pitch) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value {
    ava_allocates;
    if (map_flags != CL_MAP_WRITE_INVALIDATE_REGION) {
      ava_out;
      ava_buffer(region[0] * region[1] * region[2]);
    }
  }

  void *ret = ava_execute();
  ava_metadata(ret)->size = region[0] * region[1] * region[2];
  ava_metadata(ret)->flags = map_flags;
  ava_metadata(*event)->related_buffer = ret;
#warning fix me
}
#endif

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
                               cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  ava_unsupported;

  ava_argument(mapped_ptr) {
    ava_deallocates;
#warning This use of ava_metadata is unsafe because it changes the interpretation of the value being being looked up in the metadata map. This may mean that the first metadata lookup will use the wrong pointer.
    if (ava_metadata(mapped_ptr)->flags != CL_MAP_READ) {
      ava_in;
      ava_buffer(ava_metadata(mapped_ptr)->size);
    }
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }

  ava_execute();
  ava_metadata(mapped_ptr)->size = 0;
  ava_metadata(mapped_ptr)->flags = 0;
  ava_metadata(*event)->related_buffer = NULL;
}

cl_int clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset,
                           size_t dst_offset, size_t size, cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list, cl_event *event) {
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                           cl_event *event) {
  if (blocking_read == CL_TRUE)
    ava_sync;
  else
    ava_async;
  ava_argument(ptr) {
    ava_out;
    ava_buffer(size);
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
                            size_t cb, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
  if (blocking_write == CL_TRUE)
    ava_sync;
  else
    ava_async;

  ava_argument(ptr) {
    ava_in;
    ava_buffer(cb);
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  ava_argument(kernel) ava_object_record;
  ava_argument(arg_value) {
    if (ava_is_worker && ava_metadata(kernel)->kernel_arg_is_handle[arg_index]) {
      ava_type_cast(void **);
      ava_element {
        ava_handle;
        // ava_object_record;
      }
    } else {
      ava_element ava_opaque;
    }
    ava_in;
    ava_buffer(arg_size);
  }
}

cl_int clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event) {
  ava_consumes_resource(command_rate, 1);

  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
  ava_sync;

  ava_consumes_resource(command_rate, 1);

  ava_argument(global_work_offset) {
    ava_in;
    ava_buffer(work_dim);
  }
  ava_argument(global_work_size) {
    ava_in;
    ava_buffer(work_dim);
  }
  ava_argument(local_work_size) {
    ava_in;
    ava_buffer(work_dim);
  }
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element(ava_allocates);
  }
}

cl_int clFinish(cl_command_queue command_queue) {
  //    ava_sync;
}

cl_int clFlush(cl_command_queue command_queue) { ava_flush; }

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
  }
  ava_argument(param_value_size_ret) {
    ava_out;
    ava_buffer(1);
  }
}
