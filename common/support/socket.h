#ifndef _AVA_SUPPORT_SOCKET_H_
#define _AVA_SUPPORT_SOCKET_H_
#include <netinet/in.h>
#include <absl/strings/string_view.h>

namespace ava {
namespace support {

// Return sockfd on success, and return -1 on error
int TcpSocketConnect(struct sockaddr_in* addr);

bool ResolveTcpAddr(struct sockaddr_in* addr, absl::string_view host, absl::string_view port);

} // namespace support
} // namespace ava
#endif
