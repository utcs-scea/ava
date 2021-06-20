// clang-format off
ava_name("FPGA applications on AmorphOS");
ava_version("1.0");
ava_identifier(amorphos);
ava_number(7);
ava_cflags(-I. -I../amorphos_files);
ava_libs(-L. -L../amorphos_files -lava_aos_wrapper);
ava_export_qualifier();
// clang-format on

#include <stdint.h>
#include "ava_aos_wrapper.h"
#warning the header is at `benchmark/amorphos/f1_host/ava_aos_wrapper.h`

struct aos_client_wrapper *ava_client_new(uint64_t app_id) {
  ava_return_value ava_handle;
}

void ava_client_free(struct aos_client_wrapper *client_handle) { ava_argument(client_handle) ava_handle; }

int ava_cntrlreg_write(struct aos_client_wrapper *client_handle, uint64_t addr, uint64_t value) {
  ava_argument(client_handle) ava_handle;
}

int ava_cntrlreg_read(struct aos_client_wrapper *client_handle, uint64_t addr, uint64_t *value) {
  ava_argument(client_handle) ava_handle;
  ava_argument(value) {
    ava_buffer(1);
    ava_out;
  }
}
