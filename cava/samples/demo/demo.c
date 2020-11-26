ava_name("DEMO");
ava_version("0.0.1");
ava_identifier(DEMO);
ava_number(1);
ava_cflags(-I/usr/local/cuda-10.1/include -I../headers);

#include <demo.h>

ava_begin_utility;
#include <stdio.h>
ava_end_utility;

ava_utility
int ava_test_api_impl(int x) {
    return x + 1;
}

int ava_test_api(int x) {
    ava_disable_native_call;
    if (ava_is_worker) {
        fprintf(stderr, "RECEIVED AVA_TEST_API(%d)\n", x);
        return ava_test_api_impl(x);
    }
}
