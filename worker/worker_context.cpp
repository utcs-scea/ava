#include "worker_context.h"

#include "worker.h"

static auto common_context = ava::CommonContext::instance();
static auto worker_context = ava::WorkerContext::instance();

namespace ava {
WorkerContext::WorkerContext() { init_worker(); }
}  // namespace ava
