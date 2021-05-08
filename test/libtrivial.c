#include "libtrivial.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "testing_hack.h"

struct handle_t {
  int value;
  struct struct_buffer_t *returned_buffer;
};

int function1() { return pthread_self(); }

void function2(int (*on_error)(int, void *), void *arg) {
  int ret = on_error(42, arg);
  ck_assert_int_eq(3, ret);
}

static void (*callbacks[256])(void *, char *);

void function3a(int session, void (*callback)(void *, char *)) { callbacks[session] = callback; }

void function3b(int session, void *userdata) { callbacks[session](userdata, "This is an error"); }

void write_simple_buffer(struct simple_buffer_t *buf) {
  if (buf->buffer == NULL) {
    buf->size = 10;
    return;
  }

  for (int i = 0; i < buf->size; i++) {
    buf->buffer[i] = i;
  }
}

void read_simple_buffer(struct simple_buffer_t *buf) {
  if (buf == NULL) {
    return;
  }
  for (int i = 0; i < buf->size; i++) {
    ck_assert_int_eq(1, buf->buffer[i]);
  }
}

void mutate_simple_buffer(struct simple_buffer_t *buf) {
  for (int i = 0; i < buf->size; i++) {
    buf->buffer[i] *= 3;
  }
}

void read_call_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    ck_assert_int_eq(9, buffer[i]);
  }
}

void mutate_call_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] *= 3;
  }
}

void write_call_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] = 3;
  }
}

void read_manual_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    ck_assert_int_eq(9, buffer[i]);
  }
}

void mutate_manual_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] *= 3;
  }
}

void write_manual_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] = 3;
  }
}

void *special_alloc(size_t size) { return malloc(size); }

void special_free(void *ptr) { free(ptr); }

void read_special_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    ck_assert_int_eq(9, buffer[i]);
  }
}
void mutate_special_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] *= 3;
  }
}
void write_special_buffer(int *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] = 3;
  }
}

void free_manual_buffer(int *buffer, size_t size) {}

static void *registered_buffer;

void register_buffer(void *buffer, size_t size) { registered_buffer = buffer; }
void read_buffer(void *buffer, size_t size) {
  ck_assert_ptr_eq(registered_buffer, buffer);
  ck_assert_str_eq("%################", buffer);
}
void write_buffer(void *buffer, size_t size) {
  ck_assert_ptr_eq(registered_buffer, buffer);
  memset(buffer, '#', size);
  ((char *)buffer)[size - 1] = 0;
}

int function4a(char *in) {
  ck_assert_str_eq("zbcdefg", in);
  return strlen(in);
}

int function4b(char *out) {
  if (out == NULL) return -1;
  strncpy(out, "abcdefg", 7);
  out[7] = 0;
  return strlen(out);
}

int function4c(char *inout) {
  if (inout == NULL) return -1;
  ck_assert_str_eq("*bcdefg", inout);
  strncpy(inout, "abc", 7);
  inout[7] = 0;
  return strlen(inout);
}

void *get_buffer() { return registered_buffer; }

void unregister_buffer(void *buffer) { ck_assert_ptr_eq(registered_buffer, buffer); }

void alloc_handle(struct handle_t **buffer, int value) {
  struct handle_t *ret = calloc(1, sizeof(struct handle_t));
  *buffer = ret;
  ret->value = value;
}

int use_handle(struct handle_t *buffer) { return buffer->value; }

struct struct_buffer_t *get_handle_data(struct handle_t *handle) {
  if (handle->returned_buffer == NULL) handle->returned_buffer = calloc(1, sizeof(struct struct_buffer_t));
  handle->returned_buffer->size = handle->value;
  if (handle->returned_buffer->buffer == NULL)
    handle->returned_buffer->buffer = calloc(handle->value, sizeof(*handle->returned_buffer->buffer));
  handle->returned_buffer->handle = handle;
  handle->returned_buffer->buffer[0] = random();
  return handle->returned_buffer;
}

void free_handle(struct handle_t **buffer) {
  if ((*buffer)->returned_buffer != NULL) {
    free((*buffer)->returned_buffer->buffer);
    free((*buffer)->returned_buffer);
  }
  free(*buffer);
  *buffer = NULL;
}

//! Benchmarking functions

#define MILLIS_TO_TIMEVAL(t) \
  { (time_t) t / 1000, ((suseconds_t)t % 1000) * 1000 }

#define max(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

#define min(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })

void benchmark_fill_buffer(const void *data, size_t size);

void benchmark_check_buffer(const void *data, size_t size);

void benchmark_noop(time_t execution_time, struct timeval now_time) {
  struct timeval last_time, end_time;
  struct timeval tmp = MILLIS_TO_TIMEVAL(execution_time);
  timerclear(&last_time);
  if (!timerisset(&now_time)) gettimeofday(&now_time, NULL);
  timeradd(&now_time, &tmp, &end_time);
  unsigned long loops_per_us = 100;
  unsigned long loops_per_check = execution_time * 1000 * loops_per_us;
  do {
    if (timerisset(&last_time)) {
      // Recompute loops_per_check to be more accurate and to make roughly
      // 1000 checks regardless of the value of execution_time.
      timersub(&now_time, &last_time, &tmp);
      unsigned long us = tmp.tv_sec * 1000 * 1000 + tmp.tv_usec;
      timersub(&end_time, &now_time, &tmp);
      unsigned long remaining_us = tmp.tv_sec * 1000 * 1000 + tmp.tv_usec;
      if (us != 0) {
        loops_per_us = loops_per_check / us;
        loops_per_check = loops_per_us * min(remaining_us, execution_time);
      } else {
        loops_per_check = min(loops_per_check, loops_per_us * remaining_us);
      }
    }
    gettimeofday(&last_time, NULL);
    for (volatile unsigned long i = 0; i < loops_per_check; i++) {
      // Busy loop.
    }
    gettimeofday(&now_time, NULL);
  } while (timercmp(&now_time, &end_time, <));
}

void benchmark_copy_in_transfer_buffer(void *data, size_t size, time_t execution_time) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  benchmark_check_buffer(data, size);
  benchmark_noop(execution_time, start_time);
}

void benchmark_zero_copy_in(void *data, size_t size, time_t execution_time) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  benchmark_check_buffer(data, size);
  benchmark_noop(execution_time, start_time);
}

void benchmark_zero_copy_out(void *data, size_t size, time_t execution_time) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  benchmark_fill_buffer(data, size);
  benchmark_noop(execution_time, start_time);
}

void benchmark_copy_in_shadow_buffer(void *data, size_t size, time_t execution_time) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  benchmark_check_buffer(data, size);
  benchmark_noop(execution_time, start_time);
}

void benchmark_copy_out_existing_buffer(void *data, size_t size, time_t execution_time) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  benchmark_fill_buffer(data, size);
  benchmark_noop(execution_time, start_time);
}

void *benchmark_copy_out_shadow_buffer(size_t size, time_t execution_time) {
  static void *data = NULL;
  static size_t data_size = 0;
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  if (data_size != size) {
    if (data) free(data);
    data = malloc(size);
    data_size = size;
  }
  benchmark_fill_buffer(data, size);
  benchmark_noop(execution_time, start_time);
  return data;
}

static int touch_data = 1;

void benchmark_check_buffer(const void *data, size_t size) {
  if (touch_data) {
    for (size_t i = 0; i < size; i++) {
      if (((char *)data)[i] != 42) {
        assert("Bad data during benchmark!");
      }
    }
  }
}

void benchmark_fill_buffer(const void *data, size_t size) {
  if (touch_data) {
    for (size_t i = 0; i < size; i++) {
      ((char *)data)[i] = 42;
    }
  }
}

__attribute__((constructor)) static void __setup() {
  if (getenv("FAKE_DATA_ACCESS")) touch_data = 0;
}
