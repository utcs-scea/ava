#include "fs.h"

#include <fmt/format.h>
#include <unistd.h>

#include "common/logging.h"
#include "declaration.h"
#include "fmt.h"

namespace ava {
namespace support {

std::string GetRealPath(absl::string_view path) {
  char *result = realpath(std::string(path).c_str(), nullptr);
  if (result == nullptr) {
    AVA_LOG(WARNING) << path << " is not a valid path";
    return std::string(path);
  }
  std::string result_str(result);
  free(result);
  return result_str;
}

absl::optional<int> Create(absl::string_view full_path) {
  int fd = open(std::string(full_path).c_str(),
                /* flags= */ O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC,
                /* mode=  */ __AVA_FILE_CREAT_MODE);
  if (fd == -1) {
    AVA_LOG(ERROR) << fmt::format("Create {} failed", full_path);
    return absl::nullopt;
  }
  return fd;
}

}  // namespace support
}  // namespace ava
