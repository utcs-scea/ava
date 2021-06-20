#include <stdio.h>

#include "demo.h"

int main() {
  fprintf(stderr, "RECEIVED AVA_TEST_API = %d\n", ava_test_api(9999));
  return 0;
}
