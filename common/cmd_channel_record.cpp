#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/cmd_channel_impl.hpp"
#include "common/logging.h"

#if _FILE_OFFSET_BITS != 64
#warning "command_channel_log will fail for logs larger than 2GB. Set _FILE_OFFSET_BITS=64 at build time to fix this."
#endif

struct command_channel_log {
  struct command_channel_base base;
  int fd;
  off_t read_offset;
};

struct record_command_metadata {
  size_t size;
  uint32_t flags;  // TODO: we can set flags such as invalid bit after the
                   // command is recorded.
};

struct command_private {
  off_t command_start_offset;
  off_t current_buffer_offset;
};

//!-- Record APIs

size_t command_channel_log_buffer_size(const struct command_channel *chan, size_t size) { return size; }

// TODO: Currently this implementation can only write to files because it uses
// seek a lot.
//  We will need to abstract things a bit to allow writing to a network socket.
//  One major issue is that for the network we need to send buffers BEFORE the
//  command struct not after; because we don't want to store the buffers before
//  transmitting them.

/**
 * Create a new command for sending on this channel. This will allocate space in
 * the log for this command.
 *
 * @param c The command_channel_log.
 * @param command_struct_size The size of the command struct to allocate.
 * @param data_region_size The combined size of buffers which will be attached
 * to this command.
 * @return The new command.
 */
struct command_base *command_channel_log_new_command(struct command_channel *c, size_t command_struct_size,
                                                     size_t data_region_size) {
  struct command_channel_log *chan = (struct command_channel_log *)c;
  off_t pos = lseek(chan->fd, 0, SEEK_END);  // Seek to end.

  struct record_command_metadata metadata = {command_struct_size + data_region_size, 0};
  ssize_t ret = write(chan->fd, &metadata, sizeof(struct record_command_metadata));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  struct command_base *cmd = (struct command_base *)malloc(command_struct_size);
  static_assert(sizeof(struct command_private) <= sizeof(cmd->reserved_area),
                "command_base::reserved_area is not large enough.");
  struct command_private *priv = (struct command_private *)cmd->reserved_area;

  memset(cmd, 0, command_struct_size);
  cmd->command_size = command_struct_size;
  cmd->data_region = (void *)command_struct_size;
  cmd->region_size = data_region_size;
  priv->command_start_offset = pos + sizeof(struct record_command_metadata);
  priv->current_buffer_offset = priv->command_start_offset + command_struct_size;

  return cmd;
}

/**
 * Attach a buffer to the command by writing the data to the output file
 * directly.
 * @param c The command_channel_log.
 * @param cmd The command.
 * @param buffer A buffer to attach
 * @param size The size of buffer.
 * @return The ID of the attached buffer w.r.t. this command.
 */
void *command_channel_log_attach_buffer(struct command_channel *c, struct command_base *cmd, void *buffer,
                                        size_t size) {
  struct command_channel_log *chan = (struct command_channel_log *)c;
  struct command_private *priv = (struct command_private *)cmd->reserved_area;
  off_t pos = lseek(chan->fd, priv->current_buffer_offset, SEEK_SET);
  assert(pos == priv->current_buffer_offset);
  ssize_t ret = write(chan->fd, buffer, size);
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  return (void *)(pos - priv->command_start_offset);
}

/**
 * Finalize writing a command to this log.
 * This will write the command structure itself into the output file.
 *
 * @param c The command_channel_log
 * @param cmd The command to write.
 */
void command_channel_log_send_command(struct command_channel *c, struct command_base *cmd) {
  struct command_channel_log *chan = (struct command_channel_log *)c;
  struct command_private *priv = (struct command_private *)cmd->reserved_area;
  off_t pos = lseek(chan->fd, priv->command_start_offset, SEEK_SET);
  assert(pos == priv->command_start_offset);
  (void)pos;
  ssize_t ret = write(chan->fd, cmd, cmd->command_size);
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  // Free the local copy of the command.
  free(cmd);
}

/**
 * Serialize the command to a file descriptor and convert shared memory
 * command format into socket command format.
 *
 */
ssize_t command_channel_log_transfer_command(struct command_channel_log *c, const struct command_channel *source,
                                             const struct command_base *cmd) {
  struct command_channel_log *chan = (struct command_channel_log *)c;
  ssize_t pos = lseek(chan->fd, 0, SEEK_END);  // Seek to end.
  void *cmd_data_region = command_channel_get_data_region(source, cmd);
  struct record_command_metadata metadata = {
      .size = cmd->command_size + cmd->region_size,
      .flags = 0,
  };

  LOG_DEBUG << "record command " << cmd->command_id << " size " << std::hex << metadata.size;
  ssize_t ret;
  ret = write(chan->fd, &metadata, sizeof(struct record_command_metadata));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = write(chan->fd, cmd, cmd->command_size);
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  ret = write(chan->fd, cmd_data_region, cmd->region_size);
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  return pos;
}

/**
 * Update the flags of a recorded command. The offset in the record file
 * must be non-negative. This function is exposed only if the file
 * descriptor is seekable.
 */
void command_channel_log_update_flags(struct command_channel_log *chan, ssize_t offset, uint32_t flags) {
  assert(offset >= 0);
  off_t r = lseek(chan->fd, offset + sizeof(size_t), SEEK_SET);
  assert(r == offset + sizeof(size_t));
  (void)r;
  ssize_t ret = write(chan->fd, &flags, sizeof(flags));
  if (ret == -1) {
    fprintf(stderr, "write [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

//!-- Load APIs

/**
 * Load command from the offset from the beginning of the file. If the
 * offset value is negative, it continues from the last position. The
 * offset can be non-negative only if the file descriptor is seekable.
 */
struct command_base *command_channel_log_load_command(struct command_channel_log *chan, ssize_t offset,
                                                      uint32_t *flags) {
  struct record_command_metadata metadata;
  struct command_base *cmd;

  if (offset < 0) offset = chan->read_offset;
  lseek(chan->fd, offset, SEEK_SET);

  ssize_t size = read(chan->fd, &metadata, sizeof(struct record_command_metadata));
  if (size != sizeof(struct record_command_metadata)) return NULL;
  if (flags != NULL) {
    *flags = metadata.flags;
  }
  chan->read_offset = offset + metadata.size + sizeof(struct record_command_metadata);
  cmd = (struct command_base *)malloc(metadata.size);
  // PERFORMANCE: If this read turns out to be huge and a problem we could mmap
  // instead.
  size = read(chan->fd, cmd, metadata.size);
  if (size != metadata.size) {
    free(cmd);
    return NULL;
  }

  return cmd;
}

struct command_base *command_channel_load_next_command(struct command_channel *chan) {
  return command_channel_log_load_command((struct command_channel_log *)chan, -1, NULL);
}

/**
 * Free the loaded command.
 */
void command_channel_load_free_command(struct command_channel *c, struct command_base *cmd) { free(cmd); }

/**
 * Translate a buffer_id in the recorded command into a data pointer.
 * The returned pointer will be valid until `command_channel_load_free_command`
 * is called on `cmd`.
 */
void *command_channel_load_get_buffer(const struct command_channel *chan, const struct command_base *cmd,
                                      void *buffer_id) {
  return (void *)((uintptr_t)cmd + buffer_id);
}

void *command_channel_load_get_data_region(const struct command_channel *c, const struct command_base *cmd) {
  return (void *)((uintptr_t)cmd + cmd->command_size);
}

void command_channel_log_free(struct command_channel *c) {
  struct command_channel_log *chan = (struct command_channel_log *)c;
  close(chan->fd);
}

//! Constructor

static struct command_channel_vtable command_channel_log_vtable = {
  command_channel_buffer_size : command_channel_log_buffer_size,
  command_channel_new_command : command_channel_log_new_command,
  command_channel_attach_buffer : command_channel_log_attach_buffer,
  command_channel_send_command : command_channel_log_send_command,
  command_channel_transfer_command : (void (*)(struct command_channel *, const struct command_channel *,
                                               const struct command_base *))command_channel_log_transfer_command,
  command_channel_receive_command : command_channel_load_next_command,
  command_channel_get_buffer : command_channel_load_get_buffer,
  command_channel_get_data_region : command_channel_load_get_data_region,
  command_channel_free_command : command_channel_load_free_command,
  command_channel_free : command_channel_log_free,
  command_channel_print_command : command_channel_simple_print_command
};

struct command_channel_log *command_channel_log_new(int worker_port) {
  struct command_channel_log *chan = (struct command_channel_log *)malloc(sizeof(struct command_channel_log));
  command_channel_preinitialize((struct command_channel *)chan, &command_channel_log_vtable);

  chan->read_offset = 0;

  /* open log file */
  char fname[PATH_MAX];
  sprintf(fname, "record_log_worker_%d.bin", worker_port);
  chan->fd = open(fname, O_RDWR | O_CREAT | O_TRUNC, 0600);
  unlink(fname);
  LOG_INFO << "temporary file %s created for recording" << fname;

  return chan;
}
