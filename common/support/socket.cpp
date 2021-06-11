#include "socket.h"

#include <absl/strings/numbers.h>
#include <absl/strings/string_view.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fmt/format.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <gsl/gsl>
#include <string>

#include "common/logging.h"

namespace ava {
namespace support {

namespace {
static bool ResolveHostInternal(absl::string_view host_or_ip, struct in_addr *addr) {
  // Assume host_or_ip is IP address first
  if (inet_aton(std::string(host_or_ip).c_str(), addr) == 1) {
    return true;
  }
  // Use getaddrinfo to resolve host
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags |= AI_CANONNAME;
  struct addrinfo *result;
  int ret = getaddrinfo(std::string(host_or_ip).c_str(), nullptr, &hints, &result);
  if (ret != 0) {
    if (ret != EAI_SYSTEM) {
      AVA_LOG(ERROR) << "getaddrinfo with " << host_or_ip << " failed : " << gai_strerror(ret);
    } else {
      AVA_LOG(ERROR) << "getaddrinfo with " << host_or_ip << " failed";
    }
    return false;
  }
  auto free_freeaddrinfo_result = gsl::finally([result]() { freeaddrinfo(result); });
  while (result) {
    if (result->ai_family == AF_INET) {
      struct sockaddr_in *resolved_addr = (struct sockaddr_in *)result->ai_addr;
      *addr = resolved_addr->sin_addr;
      return true;
    }
    result = result->ai_next;
  }
  return false;
}
}  // namespace

int TcpSocketConnect(struct sockaddr_in *addr) {
  int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd == -1) {
    AVA_LOG(ERROR) << "Failed to create AF_INET socket";
    return -1;
  }
  if (connect(fd, (struct sockaddr *)addr, sizeof(struct sockaddr_in)) != 0) {
    AVA_LOG(ERROR) << "Failed to connect: " << strerror(errno);
    close(fd);
    return -1;
  }
  return fd;
}

bool ResolveTcpAddr(struct sockaddr_in *addr, absl::string_view host, absl::string_view port) {
  int parsed_port;
  if (!absl::SimpleAtoi(port, &parsed_port)) {
    return false;
  }
  addr->sin_family = AF_INET;
  addr->sin_port = htons(gsl::narrow_cast<uint16_t>(parsed_port));
  return ResolveHostInternal(host, &addr->sin_addr);
}

}  // namespace support
}  // namespace ava
