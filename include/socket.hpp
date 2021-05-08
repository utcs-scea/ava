#ifndef AVA_COMMON_SOCKET_HPP_
#define AVA_COMMON_SOCKET_HPP_

#ifdef __KERNEL__

#include <linux/types.h>
#include <linux/virtio_vsock.h>

#else

#include <linux/netlink.h>
#include <netdb.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
// clang-format off
#include <netinet/in.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SOL_NETLINK
#define SOL_NETLINK 270
#else
#ifdef __cplusplus
static_assert(SOL_NETLINK == 270, "SOL_NETLINK assumption broken");
#else
_Static_assert(SOL_NETLINK == 270, "SOL_NETLINK assumption broken");
#endif
#endif

#endif

#include "devconf.h"

enum {
  CONSUME_RC_UNSPEC = 0,

  /* internal netlink messages */
  NW_NEW_WORKER,
  NW_NEW_APPLICATION,
  NW_NEW_INVOCATION,
  COMMAND_SWAP_OUT,
  COMMAND_SWAP_IN,
  COMMAND_MSG_SWAPPING,

  /* consume resource */
  CONSUME_RC_DEVICE_TIME,
  CONSUME_RC_COMMAND_RATE,
  CONSUME_RC_QAT_THROUGHPUT,
  CONSUME_RC_DEVICE_MEMORY,
};

int init_netlink_socket(struct sockaddr_nl *src_addr, struct sockaddr_nl *dst_addr);
struct nlmsghdr *init_netlink_msg(struct sockaddr_nl *dst_addr, struct msghdr *msg, size_t size);
void free_netlink_msg(struct msghdr *msg);

int init_vm_socket(struct sockaddr_vm *sa, int cid, int port);
int conn_vm_socket(int sockfd, struct sockaddr_vm *sa);
void listen_vm_socket(int listen_fd, struct sockaddr_vm *sa_listen);
int accept_vm_socket(int listen_fd, int *guest_cid);

/**
 * send_socket - Send buffer to the socket
 * @sockfd: socket file descriptor
 * @buf: the buffer to be sent
 * @size: the buffer size
 *
 * This function is lock-free, and should be protected by locks when
 * being used.
 **/
size_t send_socket(int sockfd, const void *buf, size_t size);

/**
 * recv_socket - Receive buffer from the socket
 * @sockfd: socket file descriptor
 * @buf: the buffer to contain the received data
 * @size: the data size
 *
 * This function is lock-free, and should be protected by locks when
 * being used.
 **/
size_t recv_socket(int sockfd, void *buf, size_t size);

/**
 * parseServerAddress - Get host IP and port from a given full address
 * @full_address: can either be a full IP:port (e.g. 0.0.0.0:3333) or just
 *     the port (3333)
 * @ip: returned IP address. NULL if @full_address is a port
 * @info: server info gotten by `gethostbyname`
 * @port: return port number
 */
void parseServerAddress(const char *full_address, struct hostent **info, char *ip, int *port);

int setsockopt_lowlatency(int fd);

#ifdef __cplusplus
}
#endif

#endif  // AVA_COMMON_SOCKET_HPP_
