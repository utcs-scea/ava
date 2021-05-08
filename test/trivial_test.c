//
// Created by amp on 4/2/19.
//

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "libtrivial.h"
#include "testing_hack.h"

void *test_thread(void *);

int error_handler(int, void *);

void callback1(void *arg, char *v);

void callback2(void *arg, char *v);

START_TEST(returned_struct)
struct handle_t *handle;
alloc_handle(&handle, 4);
struct struct_buffer_t *s = get_handle_data(handle);
int old = s->buffer[0];
int *old_buffer = s->buffer;
ck_assert_int_eq(0, s->buffer[1]);
struct struct_buffer_t *r = get_handle_data(handle);
ck_assert_ptr_eq(s, r);
ck_assert_ptr_eq(old_buffer, r->buffer);
ck_assert_int_ne(old, s->buffer[0]);
ck_assert_int_eq(0, s->buffer[1]);
free_handle(&handle);
END_TEST

START_TEST(buffers_with_handles)
struct handle_t *handle;
alloc_handle(&handle, 42);
ck_assert_ptr_ne(NULL, handle);
ck_assert_uint_lt(0x400, (uintptr_t)handle);  // Check for small handles.
ck_assert_int_eq(use_handle(handle), 42);
free_handle(&handle);
ck_assert_ptr_eq(NULL, handle);
END_TEST

START_TEST(buffers_simple1)
char buffer[18];
ck_assert_int_eq(function4b(buffer), 7);
ck_assert_str_eq("abcdefg", buffer);
buffer[0] = 'z';
ck_assert_int_eq(function4a(buffer), 7);
ck_assert_str_eq("zbcdefg", buffer);
buffer[0] = '*';
ck_assert_int_eq(function4c(buffer), 3);
ck_assert_str_eq("abc", buffer);
END_TEST

#define ck_assert_int_buffer_elements_eq(buffer, size, V) \
  for (int __index = 0; __index < size; __index++) ck_assert_int_eq(V, buffer[__index])

START_TEST(buffers_simple2)
const int size = 1024 * 1024;
int *buffer = calloc(sizeof(int), size);
mutate_call_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 0);
write_call_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 3);
mutate_call_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 9);
read_call_buffer(buffer, size);
free(buffer);
END_TEST

START_TEST(buffers_manual_simple)
const int size = 1024 * 1024;
int *buffer = calloc(sizeof(int), size);
mutate_manual_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 0);
write_manual_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 3);
mutate_manual_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 9);
read_manual_buffer(buffer, size);
free_manual_buffer(buffer, size);
free(buffer);
END_TEST

START_TEST(buffers_manual_reuse)
const int size = 1024 * 1024;
int *buffer = calloc(sizeof(int), size);

// Cause the worker to repeatedly reconstruct the shadow buffer
for (int i = 0; i < 4; i++) {
  // Do manual buffer stuff
  mutate_manual_buffer(buffer, size);
  write_manual_buffer(buffer, size);
  ck_assert_int_buffer_elements_eq(buffer, size, 3);
  mutate_manual_buffer(buffer, size);
  ck_assert_int_buffer_elements_eq(buffer, size, 9);
  read_manual_buffer(buffer, size);

  // Free it.
  free_manual_buffer(buffer, size);
}

free(buffer);
END_TEST

START_TEST(buffers_special_simple)
const int size = 1024 * 1024;
int *buffer = special_alloc(size * sizeof(int));
memset(buffer, 0, size * sizeof(int));
mutate_special_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 0);
write_special_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 3);
mutate_special_buffer(buffer, size);
ck_assert_int_buffer_elements_eq(buffer, size, 9);
read_special_buffer(buffer, size);
special_free(buffer);
END_TEST

START_TEST(buffers_special_reuse)
const int size = 1024 * 16;

// Cause the worker to repeatedly reconstruct the shadow buffer
for (int i = 0; i < 4; i++) {
  int *buffer = special_alloc(size * sizeof(int));
  memset(buffer, 0, size * sizeof(int));

  // Do manual buffer stuff
  mutate_special_buffer(buffer, size);
  write_special_buffer(buffer, size);
  ck_assert_int_buffer_elements_eq(buffer, size, 3);
  mutate_special_buffer(buffer, size);
  ck_assert_int_buffer_elements_eq(buffer, size, 9);
  read_special_buffer(buffer, size);

  // Free it.
  special_free(buffer);
}
END_TEST

START_TEST(buffers_struct)
int data[18];
for (int i = 0; i < 18; i++) {
  data[i] = 1;
}

struct simple_buffer_t buf;
buf.size = 18;
buf.buffer = data;
read_simple_buffer(&buf);
write_simple_buffer(&buf);
ck_assert_int_eq(18, buf.size);
ck_assert_ptr_eq(data, buf.buffer);
for (int i = 0; i < buf.size; i++) ck_assert_int_eq(i, buf.buffer[i]);
END_TEST

START_TEST(buffers_null)
ck_assert_int_eq(-1, function4b(NULL));
ck_assert_int_eq(-1, function4c(NULL));
read_simple_buffer(NULL);
struct simple_buffer_t buf;
buf.size = 1;
buf.buffer = NULL;
write_simple_buffer(&buf);
ck_assert_int_eq(10, buf.size);
ck_assert_ptr_eq(NULL, buf.buffer);
END_TEST

START_TEST(shadow_buffers_simple)
char buffer[18] = "*";

register_buffer(buffer, 18);
ck_assert_str_eq("*", buffer);
write_buffer(buffer, 18);
ck_assert_str_eq("#################", buffer);
buffer[0] = '%';
read_buffer(buffer, 18);

char *buffer2 = get_buffer();
ck_assert_ptr_eq(buffer, buffer2);

unregister_buffer(buffer);
END_TEST

START_TEST(threads_simple)
test_thread(NULL);
pthread_t t;
pthread_create(&t, NULL, test_thread, NULL);
pthread_join(t, NULL);
function1();
END_TEST

START_TEST(callbacks)
int x = 4;
function2(error_handler, &x);
ck_assert_int_eq(4, x);
function1();
END_TEST

START_TEST(callbacks_split)
int x = 42;
int y = 43;
function3a(1, callback1);
function3a(2, callback2);

function3b(1, &x);
function3b(2, &y);
END_TEST

Suite *suite_trivial(void) {
  Suite *s;

  s = suite_create("Trivial");

#if 0
    START_TCASE(buffers)
        ADD_TEST(buffers_special_reuse);
    END_TCASE

    return s;
#else
  START_TCASE(callbacks)
  ADD_TEST(callbacks);
  ADD_TEST(callbacks_split);
  END_TCASE

  START_TCASE(buffers)
  ADD_TEST(buffers_simple1);
  ADD_TEST(buffers_simple2);
  ADD_TEST(buffers_manual_simple);
  ADD_TEST(buffers_manual_reuse);
  ADD_TEST(buffers_special_simple);
  ADD_TEST(buffers_special_reuse);
  ADD_TEST(buffers_with_handles);
  ADD_TEST(buffers_null);
  ADD_TEST(buffers_struct);
  ADD_TEST(shadow_buffers_simple);
  END_TCASE

  START_TCASE(threads)
  ADD_TEST(threads_simple);
  END_TCASE

  START_TCASE(structs)
  ADD_TEST(returned_struct);
  END_TCASE

  return s;
#endif
}

int main(int argc, char **argv) {
  int number_failed;
  Suite *s;
  s = suite_trivial();

#ifdef CHECK_MAJOR_VERSION
  SRunner *sr;
  sr = srunner_create(s);

  srunner_run_all(sr, CK_NORMAL);
  number_failed = srunner_ntests_failed(sr);
  srunner_free(sr);
  return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
#else
  return 0;
#endif
}

static int previous_tid = 0x1234;

void *test_thread(void *arg) {
  int tid = function1();
  ck_assert_int_ne(previous_tid, tid);
  previous_tid = tid;
  ck_assert_int_eq(tid, function1());
  return NULL;
}

int error_handler(int errno, void *arg) {
  ck_assert_int_eq(4, *(int *)arg);
  ck_assert_int_eq(42, errno);
  return 3;
}

static int previous_callback = 0;

void callback1(void *arg, char *v) {
  ck_assert_int_eq(0, previous_callback);
  previous_callback = 1;
  ck_assert_str_eq("This is an error", v);
  ck_assert_int_eq(42, *(int *)arg);
}

void callback2(void *arg, char *v) {
  ck_assert_int_eq(1, previous_callback);
  ck_assert_str_eq("This is an error", v);
  ck_assert_int_eq(43, *(int *)arg);
}
