//
// Created by amp on 4/13/19.
//

#ifndef AVA_TESTING_HACK_H
#define AVA_TESTING_HACK_H

#if 1
typedef struct Suite Suite;

static Suite *suite_create(const char *name) { return NULL; }

#define START_TEST(test) void test() {
#define END_TEST }

#define ck_assert_int_eq(x, y) assert(x == y)
#define ck_assert_int_ne(x, y) assert(x != y)

#define ck_assert_uint_eq(x, y) assert(x == y)
#define ck_assert_uint_ne(x, y) assert(x != y)
#define ck_assert_uint_lt(x, y) assert(x < y)
#define ck_assert_uint_gt(x, y) assert(x > y)

#define ck_assert_ptr_eq(x, y) assert(x == y)
#define ck_assert_ptr_ne(x, y) assert(x != y)

#define ck_assert_str_eq(x, y) assert(strcmp(x, y) == 0)

#define START_TCASE(name) \
  {                       \
    const char *tcase_name = #name;

#define ADD_TEST(name)                           \
  printf("         %s:%s\n", tcase_name, #name); \
  name();                                        \
  printf("++++++++ %s:%s\n", tcase_name, #name)

#define END_TCASE }

#else

#include <check.h>

#define START_TCASE(name)            \
  {                                  \
    TCase *tc = tcase_create(#name); \
    tcase_set_timeout(tc, 10);

#define ADD_TEST(name) tcase_add_test(tc, name)

#define END_TCASE         \
  suite_add_tcase(s, tc); \
  }

#endif

#endif  // AVA_TESTING_HACK_H
