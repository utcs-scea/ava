#ifndef _AVA_COMMON_SUPPORT_ENV_VARIABLE_H_
#define _AVA_COMMON_SUPPORT_ENV_VARIABLE_H_

#include <absl/strings/numbers.h>
#include <absl/strings/string_view.h>

namespace ava {
namespace support {

inline absl::string_view GetEnvVariable(absl::string_view name, absl::string_view default_value = "") {
  char *value = getenv(std::string(name).c_str());
  return value != nullptr ? value : default_value;
}

template <class IntType = int>
IntType GetEnvVariableAsInt(std::string_view name, IntType default_value = 0) {
  char *value = getenv(std::string(name).c_str());
  if (value == nullptr) {
    return default_value;
  }
  IntType result;
  if (!absl::SimpleAtoi(value, &result)) {
    return default_value;
  }
  return result;
}

}  // namespace support
}  // namespace ava

#endif  // _AVA_COMMON_SUPPORT_ENV_VARIABLE_H_
