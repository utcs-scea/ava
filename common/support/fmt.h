#ifndef _AVA_COMMON_SUPPORT_FMT_H_
#define _AVA_COMMON_SUPPORT_FMT_H_
// adopt from envy

#include <absl/strings/string_view.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace fmt {

// Provide an implementation of formatter for fmt::format that allows absl::string_view to be
// formatted with the same format specifiers available to std::string.
// NOLINTNEXTLINE(readability-identifier-naming)
template <>
struct formatter<absl::string_view> : formatter<string_view> {
  auto format(absl::string_view absl_string_view, fmt::format_context &ctx) -> decltype(ctx.out()) {
    string_view fmt_string_view(absl_string_view.data(), absl_string_view.size());
    return formatter<string_view>::format(fmt_string_view, ctx);
  }
};

}  // namespace fmt

#endif  // _AVA_COMMON_SUPPORT_FMT_H_
