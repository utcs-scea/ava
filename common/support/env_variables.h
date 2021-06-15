#ifndef _AVA_COMMON_SUPPORT_ENV_VARIABLES_H_
#define _AVA_COMMON_SUPPORT_ENV_VARIABLES_H_

#include <absl/strings/numbers.h>
#include <absl/strings/string_view.h>

namespace ava {
namespace support {

inline absl::string_view GetEnvVariable(absl::string_view name, absl::string_view default_value = "") {
  char *value = getenv(std::string(name).c_str());
  return value != nullptr ? value : default_value;
}

}  // namespace support
}  // namespace ava

#endif  // _AVA_COMMON_SUPPORT_ENV_VARIABLES_H_
