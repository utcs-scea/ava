#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <plog/Log.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/devconf.h"
#include "common/guest_mem.h"
#include "common/socket.hpp"
#include "worker.h"

// TODO: This file should be merged with cmd_channel_shm.c!!!

struct command_channel_shm {
  struct command_channel_base base;
  int guestlib_fd;
  int listen_fd;
  int shm_fd;

  struct pollfd pfd;
  MemoryRegion shm;
  struct param_block param_block;

  int vm_id;
  int listen_port;
  uint8_t init_command_type;

  /* Channel locks */
  pthread_mutex_t send_mutex;
  pthread_mutex_t recv_mutex;
};

namespace {
extern struct command_channel_vtable command_channel_shm_vtable;
}

pthread_spinlock_t block_lock;

/**
 * Print a command for debugging.
 */
static void command_channel_shm_print_command(const struct command_channel *chan, const struct command_base *cmd) {
  DEBUG_PRINT_COMMAND(chan, cmd);
}

//! Sending

/**
 * Compute the buffer size that will actually be used for a buffer of
 * `size`. The returned value may be larger than `size`.
 */
static size_t command_channel_shm_buffer_size(const struct command_channel *chan, size_t size) {
  // For shared memory implementations this should round the size up
  // to a cache line, so as to maintain the alignment of buffers when
  // they are concatinated into the data region.

  // TODO: alignment (round up to command_channel_shm->alignment)
  return size;
}

/**
 * Reserve a memory region on BAR
 *
 * Return the offset of the region or NULL if no enough space.
 * Guestlib->worker communication uses the first half of the space, and
 * the reverse communication uses the second half.
 *
 * @size: the size of the memory region.
 */
static uintptr_t reserve_param_block(struct param_block *block, size_t size) {
  uintptr_t ret_offset;

  // TODO: implement the **real** memory allocator (mask used regions)
  pthread_spin_lock(&block_lock);

  if (block->cur_offset + size >= block->size) block->cur_offset = (block->size / 2);

  ret_offset = (uintptr_t)block->cur_offset;
  block->cur_offset += size;

  pthread_spin_unlock(&block_lock);
  return ret_offset;
}

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
static struct command_base *command_channel_shm_new_command(struct command_channel *c, size_t command_struct_size,
                                                            size_t data_region_size) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;
  struct command_base *cmd = (struct command_base *)malloc(command_struct_size);
  static_assert(sizeof(struct block_seeker) <= sizeof(cmd->reserved_area),
                "command_base::reserved_area is not large enough.");
  struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

  memset(cmd, 0, command_struct_size);
  cmd->command_size = command_struct_size;
  if (data_region_size) {
    data_region_size += 0x4;
    seeker->local_offset = reserve_param_block(&chan->param_block, data_region_size);
    seeker->cur_offset = seeker->local_offset + 0x4;
    // TODO: Should the line below have `+ chan->param_block.offset`
    cmd->data_region = (void *)seeker->local_offset;
  }
  cmd->region_size = data_region_size;
  cmd->vm_id = chan->vm_id;

  return cmd;
}

/**
 * Attach a buffer to a command and return a location independent
 * buffer ID. `buffer` must be valid until after the call to
 * `command_channel_send_command`.
 *
 * The combined attached buffers must fit within the initially
 * provided `data_region_size` (to `command_channel_new_command`).
 */
static void *command_channel_shm_attach_buffer(struct command_channel *c, struct command_base *cmd, void *buffer,
                                               size_t size) {
  assert(buffer && size != 0);

  struct command_channel_shm *chan = (struct command_channel_shm *)c;
  struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
  void *offset = (void *)(seeker->cur_offset - seeker->local_offset);
  seeker->cur_offset += size;
  void *dst = (void *)((uintptr_t)chan->param_block.base + seeker->local_offset + (uintptr_t)offset);
  memcpy(dst, buffer, size);

  return offset;
}

/**
 * Send the message and all its attached buffers.
 *
 * This call is asynchronous and does not block for the command to
 * complete execution.
 */
static void command_channel_shm_send_command(struct command_channel *c, struct command_base *cmd) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;

  command_channel_shm_print_command(c, cmd);

  /* vsock interposition does not block send_message */
  pthread_mutex_lock(&chan->send_mutex);
  send_socket(chan->guestlib_fd, cmd, cmd->command_size);
  pthread_mutex_unlock(&chan->send_mutex);

  // Free local copy of command struct
  free(cmd);
}

static void command_channel_shm_transfer_command(struct command_channel *c, const struct command_channel *source,
                                                 const struct command_base *cmd) {
  struct command_base *new_cmd = command_channel_new_command(c, cmd->command_size, cmd->region_size);
  new_cmd->api_id = cmd->api_id;
  new_cmd->command_id = cmd->command_id;
  new_cmd->command_type = cmd->command_type;
  new_cmd->flags = cmd->flags;
  new_cmd->vm_id = cmd->vm_id;

  void *cmd_data_region = command_channel_get_data_region(source, cmd);
  // This call relies on the fact that command_channel_shm_attach_buffer with
  // one large buffer is identical to several calls with smaller buffers.
  command_channel_attach_buffer(c, new_cmd, cmd_data_region, cmd->region_size);
  command_channel_send_command(c, new_cmd);
}

//! Receiving

/**
 * Receive a command from a channel. The returned Command pointer
 * should be interpreted based on its `command_id` field.
 *
 * This call blocks waiting for a command to be sent along this
 * channel.
 */
static struct command_base *command_channel_shm_receive_command(struct command_channel *c) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;
  struct command_base cmd_base;
  struct command_base *cmd;
  ssize_t ret;

  ret = poll(&chan->pfd, 1, -1);
  if (ret < 0) {
    fprintf(stderr, "failed to poll\n");
    exit(-1);
  }

  if (chan->pfd.revents == 0) return NULL;

  /* terminate worker when guestlib exits */
  if (chan->pfd.revents & POLLRDHUP) {
    fprintf(stderr, "[worker#%d] guestlib shutdown\n", chan->listen_port);
    close(chan->pfd.fd);
    exit(-1);
  }

  if (chan->pfd.revents & POLLIN) {
    pthread_mutex_lock(&chan->recv_mutex);
    LOG_DEBUG << "[worker#" << chan->listen_port << "] start to recv guestlib message";
    recv_socket(chan->guestlib_fd, &cmd_base, sizeof(struct command_base));
    LOG_DEBUG << "[worker#" << chan->listen_port << "] recv guestlib message";
    cmd = (struct command_base *)malloc(cmd_base.command_size);
    memcpy(cmd, &cmd_base, sizeof(struct command_base));
    recv_socket(chan->guestlib_fd, (void *)cmd + sizeof(struct command_base),
                cmd_base.command_size - sizeof(struct command_base));
    pthread_mutex_unlock(&chan->recv_mutex);

    command_channel_shm_print_command(c, cmd);
    return cmd;
  }

  return NULL;
}

/**
 * Translate a buffer_id (as returned by
 * `command_channel_attach_buffer` in the sender) into a data pointer.
 * The returned pointer will be valid until
 * `command_channel_free_command` is called on `cmd`.
 */
static void *command_channel_shm_get_buffer(const struct command_channel *c, const struct command_base *cmd,
                                            void *buffer_id) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;
  struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
  if (buffer_id)
    return (void *)((uintptr_t)chan->param_block.base + seeker->local_offset + (uintptr_t)buffer_id);
  else
    return NULL;
}

/**
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
static void *command_channel_shm_get_data_region(const struct command_channel *c, const struct command_base *cmd) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;
  struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
  return (void *)((uintptr_t)chan->param_block.base + seeker->local_offset);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
static void command_channel_shm_free_command(struct command_channel *c, struct command_base *cmd) { free(cmd); }

/**
 * Initialize a new command channel for worker with vsock as doorbell and
 * shared memory as data transport.
 */
struct command_channel *command_channel_shm_worker_new(int listen_port) {
  struct command_channel_shm *chan = (struct command_channel_shm *)malloc(sizeof(struct command_channel_shm));
  command_channel_preinitialize((struct command_channel *)chan, &command_channel_shm_vtable);
  pthread_spin_init(&block_lock, 0);
  pthread_mutex_init(&chan->send_mutex, NULL);
  pthread_mutex_init(&chan->recv_mutex, NULL);

  /* set up worker info */
  chan->shm.size = AVA_HOST_SHM_SIZE;

  // TODO: notify executor when VM created or destroyed
  chan->listen_port = listen_port;
  assert(nw_worker_id == 0);  // TODO: Move assignment to nw_worker_id out of
                              // unrelated constructor.
  nw_worker_id = listen_port;

  /* setup shared memory */
  if ((chan->shm_fd = open("/dev/kvm-vgpu", O_RDWR | O_NONBLOCK)) < 0) {
    printf("failed to open /dev/kvm-vgpu\n");
    exit(0);
  }
  chan->shm.addr = mmap(NULL, chan->shm.size, PROT_READ | PROT_WRITE, MAP_SHARED, chan->shm_fd, 0);
  if (chan->shm.addr == MAP_FAILED) {
    printf("mmap shared memory failed: %s\n", strerror(errno));
    // TODO: add exit labels
    exit(0);
  } else
    printf("mmap shared memory to 0x%lx\n", (uintptr_t)chan->shm.addr);

  /* connect guestlib */
  struct sockaddr_vm sa_listen;
  chan->listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, chan->listen_port);
  listen_vm_socket(chan->listen_fd, &sa_listen);

  printf("[worker&%d] waiting for guestlib connection\n", listen_port);
  chan->guestlib_fd = accept_vm_socket(chan->listen_fd, NULL);
  printf("[worker@%d] guestlib connection accepted\n", listen_port);

  struct command_handler_initialize_api_command init_msg;
  recv_socket(chan->guestlib_fd, &init_msg, sizeof(struct command_handler_initialize_api_command));
  chan->init_command_type = init_msg.new_api_id;
  chan->vm_id = init_msg.base.vm_id;
  /* worker uses the last half of the parameter block.
   *   base: start address of the whole parameter block;
   *   size: size of the block;
   *   offset: offset of the block to the VM's shared memory base;
   *   cur_offset: the moving pointer for attaching buffers. */
  chan->param_block.offset = init_msg.pb_info.param_local_offset;
  chan->param_block.size = init_msg.pb_info.param_block_size;
  chan->param_block.cur_offset = (chan->param_block.size >> 1);
  chan->param_block.base = chan->shm.addr + (chan->vm_id - 1) * AVA_GUEST_SHM_SIZE + chan->param_block.offset;
  printf("[worker@%d] vm_id=%d, api_id=%x, pb_info={%lx,%lx}\n", listen_port, chan->vm_id, chan->init_command_type,
         chan->param_block.size, chan->param_block.offset);

  if (ioctl(chan->shm_fd, KVM_NOTIFY_NEW_WORKER, (unsigned long)chan->vm_id) < 0) {
    printf("failed to notify worker id\n");
    exit(0);
  }
  printf("[worker#%d] kvm-vgpu notified\n", chan->vm_id);

  // TODO: also poll netlink socket, and put the swapping task in the same
  // task queue just as the normal invocations.
  chan->pfd.fd = chan->guestlib_fd;
  chan->pfd.events = POLLIN | POLLRDHUP;

  /*
  if (fcntl(ex_st.client_fd, F_SETFL,
            fcntl(ex_st.client_fd, F_GETFL) & (~O_NONBLOCK)) < 0) {
      perror("fcntl blocking failed");
      return 0;
  }
  */

  return (struct command_channel *)chan;
}

/**
 * Disconnect worker's command channel and free all resources associated
 * with it.
 */
static void command_channel_shm_free(struct command_channel *c) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;

  pthread_spin_destroy(&block_lock);
  pthread_mutex_destroy(&chan->send_mutex);
  pthread_mutex_destroy(&chan->recv_mutex);

  munmap(chan->shm.addr, chan->shm.size);
  if (chan->shm_fd > 0) close(chan->shm_fd);
  free(chan);
}

namespace {
struct command_channel_vtable command_channel_shm_vtable = {
    command_channel_shm_buffer_size,  command_channel_shm_new_command,      command_channel_shm_attach_buffer,
    command_channel_shm_send_command, command_channel_shm_transfer_command, command_channel_shm_receive_command,
    command_channel_shm_get_buffer,   command_channel_shm_get_data_region,  command_channel_shm_free_command,
    command_channel_shm_free,         command_channel_shm_print_command};
}
