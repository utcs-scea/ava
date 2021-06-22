#ifndef _AVA_COMMON_SUPPORT_FS_H_
#define _AVA_COMMON_SUPPORT_FS_H_

#include <absl/strings/string_view.h>
#include <absl/types/optional.h>

#include <string>

namespace ava {
namespace support {

std::string GetRealPath(absl::string_view path);
// Return fd on success
absl::optional<int> Create(absl::string_view full_path);

}  // namespace support
}  // namespace ava

#endif  // _AVA_SUPPORT_FS_H_
