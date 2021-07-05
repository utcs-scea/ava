#ifndef _AVA_COMMON_SUPPORT_IO_H_
#define _AVA_COMMON_SUPPORT_IO_H_
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

#include "std_span.h"

namespace ava {
namespace support {

inline bool SendData(int fd, const char *data, size_t size) {
  size_t pos = 0;
  while (pos < size) {
    ssize_t nwrite = write(fd, data + pos, size - pos);
    if (nwrite < 0) {
      if (errno == EAGAIN || errno == EINTR) {
        continue;
      }
      return false;
    }
    pos += static_cast<size_t>(nwrite);
  }
  return true;
}

inline bool SendData(int fd, std::span<const char> data) { return SendData(fd, data.data(), data.size()); }

inline bool WriteData(int fd, std::span<const char> data) { return SendData(fd, data.data(), data.size()); }

inline bool WriteData(int fd, const char *data, size_t size) { return SendData(fd, data, size); }

inline bool WriteString(int fd, std::string data) { return SendData(fd, data.data(), data.size()); }

inline bool RecvData(int fd, char *buffer, size_t size, bool *eof) {
  size_t pos = 0;
  if (eof != nullptr) {
    *eof = false;
  }
  while (pos < size) {
    ssize_t nread = read(fd, buffer + pos, size - pos);
    if (nread == 0) {
      if (eof != nullptr) {
        *eof = true;
      }
      return false;
    }
    if (nread < 0) {
      if (errno == EAGAIN || errno == EINTR) {
        continue;
      }
      return false;
    }
    pos += static_cast<size_t>(nread);
  }
  return true;
}
const auto ReadData = RecvData;

}  // namespace support
}  // namespace ava

#endif  // _AVA_COMMON_SUPPORT_IO_H_
