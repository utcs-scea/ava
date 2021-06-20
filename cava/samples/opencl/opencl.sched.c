// clang-format off
ava_name("OpenCL");
ava_version("1.2");
ava_identifier(CL);
ava_number(2);
ava_libs(-lOpenCL);
ava_export_qualifier(CL_API_ENTRY);
// clang-format on

ava_functions ava_defaults { ava_sync; }

ava_non_transferable_types { ava_handle; }

#include <CL/cl.h>

ava_begin_utility;
#include <sys/time.h>
#include "common/logging.h"
ava_end_utility;

ava_type(cl_int) { ava_success(CL_SUCCESS); }

typedef struct {
  size_t size;

  /* argument types */
  int kernel_argc;
  char kernel_arg_is_handle[64];
} Metadata;

ava_register_metadata(Metadata);

ava_throughput_resource device_time;

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

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  ava_argument(platforms) {
    ava_out;
    cl_in_out_buffer(num_entries, num_platforms);
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
  ava_return_value ava_allocates;
}

cl_int clReleaseProgram(cl_program program) {
  ava_argument(program) { ava_deallocates; }
}

cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t *origin,
                           const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr,
                           cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  ava_unsupported;
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
  ava_argument(devices) {
    ava_out;
    cl_in_out_buffer(num_entries, num_devices);
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
  ava_return_value ava_allocates;
}

cl_context clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                                   cl_int *errcode_ret) {
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
  ava_return_value ava_allocates;
}

cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  ava_argument(param_value) {
    ava_out;
    cl_in_out_buffer(param_value_size, param_value_size_ret);
    if (ava_is_worker && param_name == CL_CONTEXT_DEVICES) {
      ava_type_cast(void **);
      ava_element { ava_handle; }
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
  ava_argument(context) { ava_deallocates; }
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value ava_allocates;
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  ava_argument(command_queue) { ava_deallocates; }
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
  ava_return_value ava_allocates;

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
  ava_argument(kernel) { ava_deallocates; }
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void (*pfn_notify)(cl_program, void *), void *user_data) {
  ava_argument(device_list) {
    ava_in;
    ava_buffer(num_devices);
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
    free((void *)options);
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

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
  ava_argument(host_ptr) {
    ava_in;
    ava_buffer(size);
  }
  ava_argument(errcode_ret) {
    ava_out;
    ava_buffer(1);
  }
}

cl_int clReleaseMemObject(cl_mem memobj) {}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
                     const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
  ava_unsupported;
}

cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t image_width,
                       size_t image_height, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
  ava_unsupported;
}

cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
                                  size_t src_offset, const size_t *dst_origin, const size_t *region,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  ava_unsupported;
}

cl_int clGetSupportedImageFormats(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type,
                                  cl_uint num_entries, cl_image_format *image_formats, cl_uint *num_image_formats) {
  ava_unsupported;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  ava_argument(event_list) {
    ava_in;
    ava_buffer(num_events);
  }
}

cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
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
  ava_argument(user_data) { ava_userdata; }
  if (ava_is_guest) clReleaseEvent(event);
}

cl_int clSetEventCallback(cl_event event, cl_int command_exec_callback_type,
                          void (*pfn_notify)(cl_event, cl_int, void *), void *user_data) {
  ava_argument(pfn_notify) { ava_callback(clSetEventCallback_callback); }
  ava_argument(user_data) { ava_userdata; }
  if (ava_is_worker) clRetainEvent(event);
  ava_execute();
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
    ava_element ava_allocates;
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
    ava_element ava_allocates;
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
    ava_element ava_allocates;
  }
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  ava_argument(arg_value) {
    if (ava_is_worker && ava_metadata(kernel)->kernel_arg_is_handle[arg_index]) {
      ava_type_cast(void **);
      ava_element { ava_handle; }
    } else {
      ava_element ava_opaque;
    }
    ava_in;
    ava_buffer(arg_size);
  }
}

ava_utility void ava_pre_execute_time(struct timeval *tv_start) { gettimeofday(tv_start, NULL); }

/* return usec */
ava_utility unsigned long ava_post_execute_time(struct timeval *tv_start) {
  struct timeval tv_end;
  gettimeofday(&tv_end, NULL);
  return (tv_end.tv_sec - tv_start->tv_sec) * 1e6 + (tv_end.tv_usec - tv_start->tv_usec);
}

cl_int clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event) {
  ava_argument(event_wait_list) {
    ava_in;
    ava_buffer(num_events_in_wait_list);
  }
  ava_argument(event) {
    ava_out;
    ava_buffer(1);
    ava_element ava_allocates;
  }

  struct timeval tv_start;
  long exec_time;
  if (ava_is_worker) ava_pre_execute_time(&tv_start);
  ava_execute();
  if (ava_is_worker) {
    clFinish(command_queue);
    exec_time = ava_post_execute_time(&tv_start);
  }
  ava_consumes_resource(device_time, exec_time);
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
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
    ava_element ava_allocates;
  }

  struct timeval tv_start;
  long exec_time;
  if (ava_is_worker) ava_pre_execute_time(&tv_start);
  ava_execute();
  if (ava_is_worker) {
    clFinish(command_queue);
    exec_time = ava_post_execute_time(&tv_start);
  }
  ava_consumes_resource(device_time, exec_time);
}

cl_int clFinish(cl_command_queue command_queue);

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
