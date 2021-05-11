#include <assert.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/devconf.h"
#include "common/guest_mem.h"
#include "common/ioctl.h"
#include "common/logging.h"
#include "common/socket.hpp"
#include "memory.h"

// TODO: This file should be merged with cmd_channel_shm_worker.c!!!

extern int vm_id;
extern struct param_block_info nw_global_pb_info;
extern int nw_global_vm_id;

struct command_channel_shm {
  struct command_channel_base base;
  int sock_fd;
  int shm_fd;
  struct pollfd pfd;
  // struct desc_slab desc_slab_list;
  struct param_block param_block;
  int vm_id;

  /* Channel locks */
  pthread_mutex_t send_mutex;
  pthread_mutex_t recv_mutex;
};

pthread_spinlock_t block_lock;

namespace {
extern struct command_channel_vtable command_channel_shm_vtable;
}

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
  // they are concatenated into the data region.

  // TODO: alignment (round up to command_channel_shm->alignment)
  return size;
}

/**
 * Reserve a memory region on BAR.
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

  if (block->cur_offset + size >= (block->size / 2)) block->cur_offset = 0;

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
    cmd->data_region = (void *)(seeker->local_offset + chan->param_block.offset);
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

  cmd->command_type = NW_NEW_INVOCATION;

  /* vsock interposition does not block send_message */
  pthread_mutex_lock(&chan->send_mutex);
  send_socket(chan->sock_fd, cmd, cmd->command_size);
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
    exit(0);
  }

  if (chan->pfd.revents == 0) return NULL;

  /* terminate guestlib when worker exits */
  if (chan->pfd.revents & POLLRDHUP) {
    AVA_WARNING << "worker shutdown";
    close(chan->pfd.fd);
    exit(0);
  }

  if (chan->pfd.revents & POLLIN) {
    pthread_mutex_lock(&chan->recv_mutex);
    memset(&cmd_base, 0, sizeof(struct command_base));
    recv_socket(chan->pfd.fd, &cmd_base, sizeof(struct command_base));
    cmd = (struct command_base *)malloc(cmd_base.command_size);
    memcpy(cmd, &cmd_base, sizeof(struct command_base));
    recv_socket(chan->pfd.fd, (void *)cmd + sizeof(struct command_base),
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
static void command_channel_shm_free_command(struct command_channel *chan, struct command_base *cmd) { free(cmd); }

/**
 * Initialize a new command channel with vsock as doorbell and shared
 * memory as data transport.
 */
struct command_channel *command_channel_shm_guest_new() {
  struct command_channel_shm *chan = (struct command_channel_shm *)malloc(sizeof(struct command_channel_shm));
  command_channel_preinitialize((struct command_channel *)chan, &command_channel_shm_vtable);
  pthread_spin_init(&block_lock, 0);
  pthread_mutex_init(&chan->send_mutex, NULL);
  pthread_mutex_init(&chan->recv_mutex, NULL);

  /* setup shared memory */
  char dev_filename[32];
  sprintf(dev_filename, "/dev/%s%d", VGPU_DEV_NAME, VGPU_DRIVER_MINOR);

  chan->shm_fd = open(dev_filename, O_RDWR);
  if (chan->shm_fd < 0) {
    fprintf(stderr, "failed to open device %s\n", dev_filename);
    exit(-1);
  }

  /* acquire vm id */
  chan->vm_id = nw_global_vm_id = ioctl(chan->shm_fd, IOCTL_GET_VM_ID);
  if (chan->vm_id <= 0) {
    fprintf(stderr, "failed to retrieve vm id: %d\n", chan->vm_id);
    exit(-1);
  }
  fprintf(stderr, "assigned vm_id=%d\n", chan->vm_id);

  chan->param_block.size = AVA_APP_SHM_SIZE_DEFAULT;
  chan->param_block.offset = ioctl(chan->shm_fd, IOCTL_REQUEST_SHM, chan->param_block.size);
  chan->param_block.base = mmap(NULL, chan->param_block.size, PROT_READ | PROT_WRITE, MAP_SHARED, chan->shm_fd, 0);
  nw_global_pb_info.param_local_offset = chan->param_block.offset;
  nw_global_pb_info.param_block_size = chan->param_block.size;
  fprintf(stderr, "param_block size=%lx, offset=%lx, base=%lx\n", chan->param_block.size, chan->param_block.offset,
          (uintptr_t)chan->param_block.base);

  /**
   * Get manager's host address from ENV('AVA_MANAGER_ADDR').
   * The address can either be a full IP:port (e.g. 0.0.0.0:3333),
   * or only the port (3333), but the IP address is always ignored as
   * the manager is assumed to be on the local server.
   */
  char *manager_full_address;
  int manager_port;
  manager_full_address = getenv("AVA_MANAGER_ADDR");
  assert(manager_full_address != NULL && "AVA_MANAGER_ADDR is not set");
  parseServerAddress(manager_full_address, NULL, NULL, &manager_port);
  assert(manager_port > 0 && "Invalid manager port");

  /* connect worker manager and send vm_id, param_block offset (inside
   * the VM's shared memory region) and param_block size. */
  struct sockaddr_vm sa;
  int manager_fd = init_vm_socket(&sa, VMADDR_CID_HOST, manager_port);
  conn_vm_socket(manager_fd, &sa);

  struct command_base *msg =
      command_channel_shm_new_command((struct command_channel *)chan, sizeof(struct command_base), 0);
  msg->command_type = NW_NEW_APPLICATION;
  struct param_block_info *pb_info = (struct param_block_info *)msg->reserved_area;
  pb_info->param_local_offset = chan->param_block.offset;
  pb_info->param_block_size = chan->param_block.size;
  send_socket(manager_fd, msg, sizeof(struct command_base));

  recv_socket(manager_fd, msg, sizeof(struct command_base));
  uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
  assert(nw_worker_id == 0);  // TODO: Move assignment to nw_worker_id out of
                              // unrelated constructor.
  nw_worker_id = worker_port;
  command_channel_shm_free_command((struct command_channel *)chan, msg);
  close(manager_fd);

  /* connect worker */
  fprintf(stderr, "assigned worker at %lu\n", worker_port);
  chan->sock_fd = init_vm_socket(&sa, VMADDR_CID_HOST, worker_port);
  // FIXME: connect is always non-blocking for vm socket!
  if (!getenv("AVA_WPOOL") || !strcmp(getenv("AVA_WPOOL"), "FALSE")) usleep(5000000);
  conn_vm_socket(chan->sock_fd, &sa);

  chan->pfd.fd = chan->sock_fd;
  chan->pfd.events = POLLIN | POLLRDHUP;

  return (struct command_channel *)chan;
}

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
static void command_channel_shm_free(struct command_channel *c) {
  struct command_channel_shm *chan = (struct command_channel_shm *)c;

  pthread_spin_destroy(&block_lock);
  pthread_mutex_destroy(&chan->send_mutex);
  pthread_mutex_destroy(&chan->recv_mutex);

  munmap(chan->param_block.base, chan->param_block.size);
  // TODO: unmap slabs
  // TODO: destroy sems

  close(chan->sock_fd);
  close(chan->shm_fd);
  free(chan);
}

namespace {
struct command_channel_vtable command_channel_shm_vtable = {
    command_channel_shm_buffer_size,  command_channel_shm_new_command,      command_channel_shm_attach_buffer,
    command_channel_shm_send_command, command_channel_shm_transfer_command, command_channel_shm_receive_command,
    command_channel_shm_get_buffer,   command_channel_shm_get_data_region,  command_channel_shm_free_command,
    command_channel_shm_free,         command_channel_shm_print_command};
}

// warning TODO: Does there need to be a separate socket specific function which
// handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the
// command_channel_socket_new and other socket specific APIs.
