#include <errno.h>
#include <string.h>

#include "common/cmd_channel.hpp"
#include "common/cmd_handler.hpp"
#include "common/declaration.h"
#include "common/devconf.h"
#include "common/socket.hpp"

struct command_channel_hv {
  int netlink_fd;
  struct sockaddr_nl src_addr;
  struct sockaddr_nl dst_addr;
  struct nlmsghdr *nlh;
  struct msghdr *nl_msg;
};

//! Utilities

void command_channel_hv_report_storage_resource_allocation(struct command_channel *AVA_UNUSED(c),
                                                           const char *const AVA_UNUSED(name),
                                                           ssize_t AVA_UNUSED(amount)) {
#if ENABLE_SWAP
  struct command_channel_hv *chan = (struct command_channel_hv *)c;
  struct command_base *raw_msg = (struct command_base *)NLMSG_DATA(chan->nlh);

  if (!strcmp(name, "device_memory")) {
    raw_msg->command_id = CONSUME_RC_DEVICE_MEMORY;
    *((long *)raw_msg->reserved_area) = (long)amount;
    sendmsg(chan->netlink_fd, chan->nl_msg, 0);
  }
#endif
}

void command_channel_hv_report_throughput_resource_consumption(struct command_channel *c, const char *const name,
                                                               ssize_t amount) {
#ifdef AVA_ENABLE_KVM_MEDIATION
  struct command_channel_hv *chan = (struct command_channel_hv *)c;
  struct command_base *raw_msg = (struct command_base *)NLMSG_DATA(chan->nlh);

#if ENABLE_REPORT_BATCH
  static uint32_t api_count = 0;
  static ssize_t total_amount = 0;

  api_count++;
  total_amount += amount;
  /* report resource when over 10 invocations are counted or the total
   * time is over 5 ms */
  if (api_count >= 20 || amount >= 5000) {
    if (!strcmp(name, "command_rate"))
      amount = api_count;
    else if (!strcmp(name, "device_time") || !strcmp(name, "qat_throughput"))
      amount = total_amount;
    api_count = total_amount = 0;
  } else {
    return;
  }
#endif

  if (!strcmp(name, "command_rate")) {
    raw_msg->command_id = CONSUME_RC_COMMAND_RATE;
    *((int *)raw_msg->reserved_area) = (int)amount;
    sendmsg(chan->netlink_fd, chan->nl_msg, 0);
  }

  if (!strcmp(name, "device_time")) {
    raw_msg->command_id = CONSUME_RC_DEVICE_TIME;
    *((long *)raw_msg->reserved_area) = (long)amount;
    sendmsg(chan->netlink_fd, chan->nl_msg, 0);
  }

  if (!strcmp(name, "qat_throughput")) {
    raw_msg->command_id = CONSUME_RC_QAT_THROUGHPUT;
    *((long *)raw_msg->reserved_area) = (long)amount;
    sendmsg(chan->netlink_fd, chan->nl_msg, 0);
  }
#endif
}

struct command_channel *command_channel_hv_new(int worker_port) {
  struct command_channel_hv *chan = (struct command_channel_hv *)malloc(sizeof(struct command_channel_hv));
  memset(chan, 0, sizeof(struct command_channel_hv));

  /* connect hypervisor */
#ifdef AVA_ENABLE_KVM_MEDIATION
  printf("establish netlink channel for worker@%d\n", worker_port);

  chan->netlink_fd = init_netlink_socket(&chan->src_addr, &chan->dst_addr);
  chan->nl_msg = (struct msghdr *)malloc(sizeof(struct msghdr));
  chan->nlh = init_netlink_msg(&chan->dst_addr, chan->nl_msg, sizeof(struct command_base));

  struct command_base *raw_msg = (struct command_base *)NLMSG_DATA(chan->nlh);
  raw_msg->api_id = COMMAND_HANDLER_API;
  raw_msg->command_id = NW_NEW_WORKER;
  *((int *)raw_msg->reserved_area) = worker_port;
  raw_msg->vm_id = 0;

  ssize_t retsize;
  retsize = sendmsg(chan->netlink_fd, chan->nl_msg, 0);
  if (retsize < 0) {
    printf("sendmsg failed with errcode %s\n", strerror(errno));
    exit(-1);
  } else
    printf("[worker#%d] kvm-vgpu netlink notified\n", worker_port);
#endif

  return (struct command_channel *)chan;
}

/**
 * Disconnect worker's netlink channel and free all resources associated
 * with it.
 */
void command_channel_hv_free(struct command_channel *c) {
  struct command_channel_hv *chan = (struct command_channel_hv *)c;

  if (chan->nl_msg) free_netlink_msg(chan->nl_msg);
  free(chan);
}
