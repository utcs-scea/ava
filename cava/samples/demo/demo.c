// clang-format off
ava_name("DEMO");
ava_version("0.0.1");
ava_identifier(DEMO);
ava_number(1);
ava_cflags(-I${CMAKE_SOURCE_DIR}/cava/headers);
ava_soname(libdemo.so);
// clang-format on

#include <demo.h>

ava_begin_utility;
#include <stdio.h>
#include "common/logging.h"
ava_end_utility;

ava_utility int ava_test_api_impl(int x) { return x + 1; }

int ava_test_api(int x) {
  ava_disable_native_call;
  if (ava_is_worker) {
    ava_info("RECEIVED AVA_TEST_API(%d)\n", x);
    return ava_test_api_impl(x);
  }
}
