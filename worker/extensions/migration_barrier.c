#include "common/extensions/migration_barrier.h"

// migration barrier not valid for the worker. This code is only here due to
// AvA's build system requiring extensions have the same interface on guestlib
// and worker. (#86)

void migration_barrier_init(void) {
}

void migration_barrier_destroy(void) {
}

void migration_barrier_wait(long long int call_id) {
}
