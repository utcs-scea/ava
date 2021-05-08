//
// Created by amp on 4/2/19.
//

#ifndef AVA_LIBTRIVIAL_H
#define AVA_LIBTRIVIAL_H

#include <stdlib.h>

struct handle_t;

struct struct_buffer_t {
  struct handle_t *handle;
  int *buffer;
  size_t size;
};

struct simple_buffer_t {
  int *buffer;
  size_t size;
};

int function1();
int function4a(char *in);
int function4b(char *out);
int function4c(char *inout);

void function2(int (*on_error)(int, void *), void *arg);
void function3a(int session, void (*callback)(void *, char *));
void function3b(int session, void *userdata);

void write_simple_buffer(struct simple_buffer_t *buf);
void mutate_simple_buffer(struct simple_buffer_t *buf);
void read_simple_buffer(struct simple_buffer_t *buf);

void read_call_buffer(int *buffer, size_t size);
void mutate_call_buffer(int *buffer, size_t size);
void write_call_buffer(int *buffer, size_t size);

void read_manual_buffer(int *buffer, size_t size);
void mutate_manual_buffer(int *buffer, size_t size);
void write_manual_buffer(int *buffer, size_t size);
void free_manual_buffer(int *buffer, size_t size);

void register_buffer(void *buffer, size_t size);
void read_buffer(void *buffer, size_t size);
void write_buffer(void *buffer, size_t size);
void *get_buffer();
void unregister_buffer(void *buffer);

void alloc_handle(struct handle_t **handle, int value);
int use_handle(struct handle_t *handle);
struct struct_buffer_t *get_handle_data(struct handle_t *handle);
void free_handle(struct handle_t **handle);

//! Zero-copy

void *special_alloc(size_t size);
void special_free(void *ptr);

// Used with special buffers only.

void read_special_buffer(int *buffer, size_t size);
void mutate_special_buffer(int *buffer, size_t size);
void write_special_buffer(int *buffer, size_t size);

//! Benchmarking functions

void benchmark_noop(time_t execution_time, struct timeval now_time);
void benchmark_zero_copy_in(void *data, size_t size, time_t execution_time);
void benchmark_zero_copy_out(void *data, size_t size, time_t execution_time);
void benchmark_copy_in_transfer_buffer(void *data, size_t size, time_t execution_time);
void benchmark_copy_in_shadow_buffer(void *data, size_t size, time_t execution_time);
void benchmark_copy_out_existing_buffer(void *data, size_t size, time_t execution_time);
void *benchmark_copy_out_shadow_buffer(size_t size, time_t execution_time);

#endif  // AVA_LIBTRIVIAL_H
