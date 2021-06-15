#ifndef _AVA_COMMON_SUPPORT_GEN_STAT_H_
#define _AVA_COMMON_SUPPORT_GEN_STAT_H_

#include <fmt/format.h>
#include <stdint.h>

#include <gsl/gsl>

#include "common/extensions/cmd_batching.h"
#include "common/support/io.h"
#include "guestlib/guest_thread.h"
#include "time_util.h"

namespace ava {
namespace support {

inline void stats_end(const char *func_name, int64_t begin_ts) {
  auto end_ts = ava::GetMonotonicNanoTimestamp();
  fmt::memory_buffer output;
  fmt::format_to(output, "GuestlibStat {}, {}\n", func_name, gsl::narrow_cast<int32_t>(end_ts - begin_ts));
  ava::guest_write_stats(output.data(), output.size());
}

}  // namespace support
}  // namespace ava

#endif  // _AVA_COMMON_SUPPORT_GEN_STAT_H_
