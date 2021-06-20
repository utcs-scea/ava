// clang-format off
ava_name("Trivial Library");
ava_version("0.1");
ava_identifier(test);
ava_number(255);
ava_cflags(-I${CMAKE_SOURCE_DIR}/test);
ava_libs(-L${CMAKE_SOURCE_DIR}/test -ltrivial);
ava_export_qualifier();
ava_soname(libtrivial.so);
// clang-format on

#include "libtrivial.h"

ava_begin_utility;
#ifndef __CAVA__
#include "common/zcopy.h"
#endif
ava_end_utility;

struct metadata_t {
  void (*session_callback)(void *, char *);
};

ava_type(struct struct_buffer_t) {
  struct struct_buffer_t *ava_self;
  ava_field(buffer) {
    ava_buffer(ava_self->size);
    ava_lifetime_coupled(ava_self->handle);
  }
}
ava_type(struct simple_buffer_t) {
  struct simple_buffer_t *ava_self;
  ava_field(buffer) {
    ava_buffer(ava_self->size);
    ava_lifetime_coupled(ava_self);
  }
}

ava_begin_utility;
static size_t registered_buffer_size;

void *my_alloc(size_t size) { return malloc(size); }

void my_free(void *ptr) { free(ptr); }

ava_end_utility;

ava_register_metadata(struct metadata_t);

//! Trivial tests

int function1() { ava_sync; }

//! Callbacks

ava_callback_decl int error_callback(int errno_, void *arg) {
  ava_argument(arg) { ava_userdata; }
}

ava_callback_decl void other_callback(void *arg, char *error) {
  ava_argument(arg) { ava_userdata; }
  ava_argument(error) {
    ava_buffer(strlen(error) + 1);
    ava_in;
  }
}

void function2(int (*on_error)(int, void *), void *arg) {
  ava_async;
  ava_argument(on_error) { ava_callback(error_callback); }
  ava_argument(arg) { ava_userdata; }
}

void function3a(int session, void (*callback)(void *, char *)) {
  ava_sync;

  ava_argument(callback) { ava_callback_registration(other_callback); }

  ava_metadata((void *)session)->session_callback = callback;
  ava_execute();
}

void function3b(int session, void *userdata) {
  ava_implicit_argument void (*callback)(void *, char *) = ava_metadata((void *)session)->session_callback;

  ava_sync;

  ava_argument(callback) { ava_callback(other_callback); }
  ava_argument(userdata) { ava_userdata; }

  ava_execute();
}

//! Buffers

int function4a(char *in) {
  ava_sync;
  ava_argument(in) {
    ava_buffer(8);
    ava_input;
  }
}

int function4b(char *out) {
  ava_sync;

  ava_argument(out) {
    ava_buffer(8);
    ava_output;
  }
}

int function4c(char *inout) {
  ava_sync;

  ava_argument(inout) {
    ava_buffer(8);
    ava_input;
    ava_output;
  }
}

//! Shadow (e.i., managed coupled) buffers.

void register_buffer(void *buffer, size_t size) {
  ava_sync;
  ava_argument(buffer) {
    ava_buffer(size);
    ava_lifetime_manual;
    ava_no_copy;
  }

  registered_buffer_size = size;
  ava_execute();
}

void read_buffer(void *buffer, size_t size) {
  ava_async;
  ava_argument(buffer) {
    ava_buffer(size);
    ava_lifetime_manual;
    ava_input;
  }

  ava_execute();
}

void write_buffer(void *buffer, size_t size) {
  ava_sync;
  ava_argument(buffer) {
    ava_buffer(size);
    ava_lifetime_manual;
    ava_output;
  }

  ava_execute();
}

void *get_buffer() {
  ava_sync;

  ava_return_value {
    ava_buffer(registered_buffer_size);
    ava_output;
    ava_lifetime_manual;
  }

  ava_execute();
}

void unregister_buffer(void *buffer) {
  ava_async;

  ava_argument(buffer) {
    ava_buffer(registered_buffer_size);
    ava_lifetime_manual;
    ava_no_copy;
    ava_deallocates;
  }

  ava_execute();
}

//! Handles

void alloc_handle(struct handle_t **buffer, int value) {
  ava_sync;

  ava_argument(buffer) {
    ava_element { ava_allocates; }
    ava_buffer(1);
    ava_output;
  }
}

int use_handle(struct handle_t *buffer) { ava_sync; }

struct struct_buffer_t *get_handle_data(struct handle_t *handle) {
  ava_sync;

  ava_argument(handle) { ava_handle; }
  ava_return_value {
    ava_buffer(1);
    ava_lifetime_coupled(handle);
    ava_out;
  }
}

void free_handle(struct handle_t **buffer) {
  ava_sync;
  ava_argument(buffer) {
    ava_buffer(1);
    ava_input;
    ava_output;
    ava_element { ava_deallocates; }
  }
}

void write_simple_buffer(struct simple_buffer_t *buf) {
  ava_sync;

  ava_argument(buf) {
    ava_buffer(1);
    ava_in;
    ava_out;
  }
}

void read_simple_buffer(struct simple_buffer_t *buf) {
  ava_async;

  ava_argument(buf) {
    ava_buffer(1);
    ava_in;
  }
}

void mutate_simple_buffer(struct simple_buffer_t *buf) {
  ava_sync;

  ava_argument(buf) {
    ava_buffer(1);
    ava_in;
    ava_out;
  }

  ava_execute();
}

void read_call_buffer(int *buffer, size_t size) {
  ava_async;

  ava_argument(buffer) {
    ava_input;
    ava_buffer(size);
  }

  ava_execute();
}

void mutate_call_buffer(int *buffer, size_t size) {
  ava_sync;

  ava_argument(buffer) {
    ava_input;
    ava_output;
    ava_buffer(size);
  }

  ava_execute();
}

void write_call_buffer(int *buffer, size_t size) {
  ava_sync;

  ava_argument(buffer) {
    ava_output;
    ava_buffer(size);
  }

  ava_execute();
}

void read_manual_buffer(int *buffer, size_t size) {
  ava_async;

  ava_argument(buffer) {
    ava_input;
    ava_buffer(size);
    ava_lifetime_manual;
  }

  ava_execute();
}

void mutate_manual_buffer(int *buffer, size_t size) {
  ava_sync;

  ava_argument(buffer) {
    ava_input;
    ava_output;
    ava_buffer(size);
    ava_lifetime_manual;
  }

  ava_execute();
}

void write_manual_buffer(int *buffer, size_t size) {
  ava_sync;

  ava_argument(buffer) {
    ava_output;
    ava_buffer(size);
    ava_lifetime_manual;
  }

  ava_execute();
}

void free_manual_buffer(int *buffer, size_t size) {
  ava_async;

  ava_argument(buffer) {
    ava_no_copy;
    ava_deallocates;
    ava_buffer(size);
    ava_lifetime_manual;
  }

  ava_execute();
}

void benchmark_noop(time_t execution_time, struct timeval now_time) { ava_sync; }

void benchmark_copy_in_transfer_buffer(void *data, size_t size, time_t execution_time) {
  ava_sync;

  ava_argument(data) {
    ava_input;
    ava_buffer(size);
  }
}

void benchmark_copy_in_shadow_buffer(void *data, size_t size, time_t execution_time) {
  ava_sync;

  ava_argument(data) {
    ava_input;
    ava_buffer(size);
    ava_lifetime_static;
  }
}

void benchmark_copy_out_existing_buffer(void *data, size_t size, time_t execution_time) {
  ava_sync;

  ava_argument(data) {
    ava_output;
    ava_buffer(size);
  }
}

void *benchmark_copy_out_shadow_buffer(size_t size, time_t execution_time) {
  ava_sync;

  ava_return_value {
    ava_output;
    ava_buffer(size);
    ava_lifetime_static;
  }
}

void benchmark_zero_copy_in(void *data, size_t size, time_t execution_time) {
  ava_sync;

  ava_argument(data) { ava_zerocopy_buffer; }
}

void benchmark_zero_copy_out(void *data, size_t size, time_t execution_time) {
  ava_sync;

  ava_argument(data) { ava_zerocopy_buffer; }
}

ava_begin_replacement;
void *special_alloc(size_t size) { return ava_zerocopy_alloc(size); }

void special_free(void *ptr) { ava_zerocopy_free(ptr); }
ava_end_replacement;

void read_special_buffer(int *buffer, size_t size) {
  ava_sync;
  ava_argument(buffer) { ava_zerocopy_buffer; }

  ava_execute();
}

void mutate_special_buffer(int *buffer, size_t size) {
  ava_sync;
  ava_argument(buffer) { ava_zerocopy_buffer; }

  ava_execute();
}

void write_special_buffer(int *buffer, size_t size) {
  ava_sync;
  ava_argument(buffer) { ava_zerocopy_buffer; }

  ava_execute();
}
