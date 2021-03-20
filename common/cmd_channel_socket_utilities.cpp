#include <assert.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <string.h>
#include <unistd.h>

#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"
#include "common/guest_mem.h"
#include "common/cmd_handler.h"
#include "common/cmd_channel_socket_utilities.hpp"

namespace chansocketutil {

/**
 * Print a command for debugging.
 */
void command_channel_socket_print_command(const struct command_channel *chan, const struct command_base *cmd)
{
    /*
    DEBUG_PRINT("struct command_base {\n"
                "  command_type=%ld\n"
                "  flags=%d\n"
                "  api_id=%d\n"
                "  command_id=%ld\n"
                "  command_size=%lx\n"
                "  region_size=%lx\n"
                "}\n",
                cmd->command_type,
                cmd->flags,
                cmd->api_id,
                cmd->command_id,
                cmd->command_size,
                cmd->region_size);
    */
    DEBUG_PRINT_COMMAND(chan, cmd);
}

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
void command_channel_socket_free(struct command_channel* c) {
    struct command_channel_socket* chan = (struct command_channel_socket*)c;
    if (chan->listen_fd)
        close(chan->listen_fd);
    close(chan->sock_fd);
    pthread_mutex_destroy(&chan->send_mutex);
    pthread_mutex_destroy(&chan->recv_mutex);
    free(chan);
}

//! Sending

/**
 * Compute the buffer size that will actually be used for a buffer of
 * `size`. The returned value may be larger than `size`.
 * For shared memory implementations this should round the size up
 * to a cache line, so as to maintain the alignment of buffers when
 * they are concatinated into the data region.
 */
size_t command_channel_socket_buffer_size(const struct command_channel *c, size_t size) {
    return size;
}

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
struct command_base* command_channel_socket_new_command(struct command_channel* c, size_t command_struct_size, size_t data_region_size) {
    struct command_channel_socket* chan = (struct command_channel_socket *)c;
    struct command_base *cmd = (struct command_base *)malloc(command_struct_size + data_region_size);
    static_assert(sizeof(struct block_seeker) <= sizeof(cmd->reserved_area),
                  "command_base::reserved_area is not large enough.");
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    memset(cmd, 0, command_struct_size);
    cmd->vm_id = chan->vm_id;
    cmd->command_size = command_struct_size;
    cmd->data_region = (void *)command_struct_size;
    cmd->region_size = data_region_size;
    seeker->cur_offset = command_struct_size;

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
void* command_channel_socket_attach_buffer(struct command_channel* c, struct command_base* cmd, void* buffer, size_t size) {
    assert(buffer && size != 0);

    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    void *offset = (void *)seeker->cur_offset;
    void *dst = (void *)((uintptr_t)cmd + seeker->cur_offset);
    seeker->cur_offset += size;
    memcpy(dst, buffer, size);
    return offset;
}

/**
 * Send the message and all its attached buffers.
 *
 * This call is asynchronous and does not block for the command to
 * complete execution.
 */
void command_channel_socket_send_command(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)c;
    cmd->command_type = NW_NEW_INVOCATION;

    /* vsock interposition does not block send_message */
    pthread_mutex_lock(&chan->send_mutex);
    send_socket(chan->sock_fd, cmd, cmd->command_size + cmd->region_size);
    pthread_mutex_unlock(&chan->send_mutex);

    // Free the local copy of the command and buffers.
    free(cmd);
}

void command_channel_socket_transfer_command(struct command_channel* c, const struct command_channel *source,
                                                    const struct command_base *cmd)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)c;
    void *cmd_data_region = command_channel_get_data_region(source, cmd);

    send_socket(chan->sock_fd, cmd, cmd->command_size);
    send_socket(chan->sock_fd, cmd_data_region, cmd->region_size);
}

//! Receiving

/**
 * Receive a command from a channel. The returned Command pointer
 * should be interpreted based on its `command_id` field.
 *
 * This call blocks waiting for a command to be sent along this
 * channel.
 */
struct command_base* command_channel_socket_receive_command(struct command_channel* c)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)c;
    struct command_base cmd_base;
    struct command_base *cmd;
    ssize_t ret;

    ret = poll(&chan->pfd, 1, -1);
    if (ret < 0) {
        fprintf(stderr, "failed to poll\n");
        exit(-1);
    }

    if (chan->pfd.revents == 0)
        return NULL;

    /* terminate guestlib when worker exits */
    if (chan->pfd.revents & POLLRDHUP) {
        DEBUG_PRINT("command_channel_socket shutdown\n");
        close(chan->pfd.fd);
        exit(-1);
    }

    if (chan->pfd.revents & POLLIN) {
        pthread_mutex_lock(&chan->recv_mutex);
        memset(&cmd_base, 0, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, &cmd_base, sizeof(struct command_base));
        cmd = (struct command_base *)malloc(cmd_base.command_size + cmd_base.region_size);
        memcpy(cmd, &cmd_base, sizeof(struct command_base));

        recv_socket(chan->pfd.fd, (uint8_t *)cmd + sizeof(struct command_base),
                    cmd_base.command_size + cmd_base.region_size - sizeof(struct command_base));
        pthread_mutex_unlock(&chan->recv_mutex);

        command_channel_socket_print_command(c, cmd);
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
void* command_channel_socket_get_buffer(const struct command_channel *chan, const struct command_base *cmd, void* buffer_id) {
    return (void *)((uintptr_t)cmd + buffer_id);
}

/**
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
void* command_channel_socket_get_data_region(const struct command_channel *c, const struct command_base *cmd)
{
    return (void *)((uintptr_t)cmd + cmd->command_size);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
void command_channel_socket_free_command(struct command_channel* c, struct command_base* cmd) {
    free(cmd);
}

};  // namespace chansocketutil
