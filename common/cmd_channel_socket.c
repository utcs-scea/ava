#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"
#include "common/guest_mem.h"
#include "common/cmd_handler.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern int nw_global_vm_id;

struct command_channel_socket {
    struct command_channel_base base;
    int sock_fd;
    struct pollfd pfd;
    uint8_t vm_id;

    /* Channel locks */
    pthread_mutex_t send_mutex;
    pthread_mutex_t recv_mutex;

    // TODO: Remove the following fields that don't seem to do anything.
    int listen_fd;
    int listen_port;
    uint8_t init_command_type;
};

static struct command_channel_vtable command_channel_socket_vtable;

/**
 * Configure fd for low-latency transmission.
 *
 * This currently sets TCP_NODELAY.
 */
static int setsockopt_lowlatency(int fd)
{
    int enabled = 1;
    int r = setsockopt(fd, SOL_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
    if(r)
        perror("setsockopt TCP_NODELAY");
    return r;
}

/**
 * Print a command for debugging.
 */
static void command_channel_socket_print_command(const struct command_channel *chan, const struct command_base *cmd)
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
static void command_channel_socket_free(struct command_channel* c) {
    struct command_channel_socket* chan = (struct command_channel_socket*)c;
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
static size_t command_channel_socket_buffer_size(const struct command_channel *c, size_t size) {
    return size;
}

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
static struct command_base* command_channel_socket_new_command(struct command_channel* c, size_t command_struct_size, size_t data_region_size) {
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
static void* command_channel_socket_attach_buffer(struct command_channel* c, struct command_base* cmd, void* buffer, size_t size) {
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
static void command_channel_socket_send_command(struct command_channel* c, struct command_base* cmd)
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

static void command_channel_socket_transfer_command(struct command_channel* c, const struct command_channel *source,
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
static struct command_base* command_channel_socket_receive_command(struct command_channel* c)
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

        recv_socket(chan->pfd.fd, (void *)cmd + sizeof(struct command_base),
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
static void * command_channel_socket_get_buffer(const struct command_channel *chan, const struct command_base *cmd, void* buffer_id) {
    return (void *)((uintptr_t)cmd + buffer_id);
}

/**
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
static void * command_channel_socket_get_data_region(const struct command_channel *c, const struct command_base *cmd)
{
    return (void *)((uintptr_t)cmd + cmd->command_size);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
static void command_channel_socket_free_command(struct command_channel* c, struct command_base* cmd) {
    free(cmd);
}

struct command_channel* command_channel_min_new()
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    chan->vm_id = nw_global_vm_id = 1;

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

    /**
     * Connect manager which shall return the assigned API server's
     * address. The address can either be a full IP:port or only the port
     * if the API server is on the same machine as the manager.
     */
    struct sockaddr_in serv_addr;
    int manager_fd = socket(AF_INET, SOCK_STREAM, 0);

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(manager_port);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
    connect(manager_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    struct command_base* msg = command_channel_socket_new_command((struct command_channel *)chan, sizeof(struct command_base), 0);
    msg->command_type = NW_NEW_APPLICATION;
    struct param_block_info *pb_info = (struct param_block_info *)msg->reserved_area;
    pb_info->param_local_offset = 0;
    pb_info->param_block_size = 0;
    send_socket(manager_fd, msg, sizeof(struct command_base));

    recv_socket(manager_fd, msg, sizeof(struct command_base));
    uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
    assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
    nw_worker_id = worker_port;
    command_channel_socket_free_command((struct command_channel *)chan, msg);
    close(manager_fd);

    /* connect worker */
    DEBUG_PRINT("assigned worker at %lu\n", worker_port);
    chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt_lowlatency(chan->sock_fd);
    serv_addr.sin_port = htons(worker_port);

    /* on mirage `connect` is always non-blocking and the server must
     * start before the guest.
     *
    int sock_flags = fcntl(chan->sock_fd, F_GETFL);
    if (sock_flags & O_NONBLOCK) {
        DEBUG_PRINT("socket was non-blocking\n");
        if (fcntl(chan->sock_fd, F_SETFL, sock_flags & (~O_NONBLOCK)) < 0) {
            perror("fcntl blocking");
            exit(0);
        }
    }
    */

    int ret;
    do {
        if (!getenv("AVA_WPOOL") || !strcmp(getenv("AVA_WPOOL"), "FALSE"))
            usleep(2000000);
        ret = connect(chan->sock_fd, (struct sockaddr *) &serv_addr, sizeof(serv_addr));
    } while(ret == -1);

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

struct command_channel* command_channel_min_worker_new(int listen_port)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    // TODO: notify executor when VM created or destroyed
    printf("spawn worker port#%d, rt_type#%x\n", listen_port, chan->init_command_type);
    chan->listen_port = listen_port;
    assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
    nw_worker_id = listen_port;

    /* connect guestlib */
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;

    if ((chan->listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket");
    }
    setsockopt_lowlatency(chan->listen_fd);
    // Forcefully attaching socket to the worker port
    if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(listen_port);

    if (bind(chan->listen_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
    }
    if (listen(chan->listen_fd, 10) < 0) {
        perror("listen");
    }

    printf("[worker@%d] waiting for guestlib connection\n", listen_port);
    chan->sock_fd = accept(chan->listen_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);

    // TODO: accept init message from guestlib
    struct command_handler_initialize_api_command init_msg;
    recv_socket(chan->sock_fd, &init_msg, sizeof(struct command_handler_initialize_api_command));
    chan->init_command_type = init_msg.new_api_id;
    chan->vm_id = init_msg.base.vm_id;
    printf("[worker@%d] vm_id=%d, api_id=%x\n", listen_port, chan->vm_id, chan->init_command_type);

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

struct command_channel* command_channel_socket_new()
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    chan->vm_id = nw_global_vm_id = 1;

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

    /* connect manager to get worker port */
    struct sockaddr_vm sa;
    int manager_fd = init_vm_socket(&sa, VMADDR_CID_HOST, manager_port);
    conn_vm_socket(manager_fd, &sa);

    struct command_base* msg = command_channel_socket_new_command((struct command_channel *)chan, sizeof(struct command_base), 0);
    msg->command_type = NW_NEW_APPLICATION;
    send_socket(manager_fd, msg, sizeof(struct command_base));

    recv_socket(manager_fd, msg, sizeof(struct command_base));
    uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
    assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
    nw_worker_id = worker_port;
    command_channel_socket_free_command((struct command_channel *)chan, msg);
    close(manager_fd);

    /* connect worker */
    DEBUG_PRINT("assigned worker at %lu\n", worker_port);
    chan->sock_fd = init_vm_socket(&sa, VMADDR_CID_HOST, worker_port);
    // FIXME: connect is always non-blocking for vm socket!
    if (!getenv("AVA_WPOOL") || !strcmp(getenv("AVA_WPOOL"), "FALSE"))
        usleep(2000000);
    conn_vm_socket(chan->sock_fd, &sa);

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

struct command_channel* command_channel_socket_worker_new(int listen_port)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    // TODO: notify executor when VM created or destroyed
    printf("spawn worker port#%d\n", listen_port);
    chan->listen_port = listen_port;
    assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
    nw_worker_id = listen_port;

    /* connect guestlib */
    struct sockaddr_vm sa_listen;
    chan->listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, chan->listen_port);
    listen_vm_socket(chan->listen_fd, &sa_listen);

    printf("[worker@%d] waiting for guestlib connection\n", listen_port);
    chan->sock_fd = accept_vm_socket(chan->listen_fd, NULL);

    struct command_handler_initialize_api_command init_msg;
    recv_socket(chan->sock_fd, &init_msg, sizeof(struct command_handler_initialize_api_command));
    chan->init_command_type = init_msg.new_api_id;
    chan->vm_id = init_msg.base.vm_id;
    printf("[worker@%d] vm_id=%d, api_id=%x\n", listen_port, chan->vm_id, chan->init_command_type);

    // TODO: also poll netlink socket, and put the swapping task in the same
    // task queue just as the normal invocations.
    chan->pfd.fd = chan->sock_fd;
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
 * TCP channel.
 * @worker_port: the listening port of worker. Only worker needs to provide
 * this argument.
 * @is_guest: 1 if it is client (guestlib), otherwise 0
 *
 * The `manager_tcp` is required to use the TCP channel.
 * */
struct command_channel *command_channel_socket_tcp_new(int worker_port, int is_guest)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    memset(&address, 0, sizeof(address));

    if (is_guest) {
        chan->vm_id = nw_global_vm_id = 1;

        /**
         * Get manager's host address from ENV('AVA_MANAGER_ADDR').
         * The address can either be a full IP:port (e.g. 0.0.0.0:3333),
         * or only the port (3333) if the manager is on the local server.
         */
        char *manager_full_address;
        char manager_name[128];
        int manager_port;
        struct hostent *server_info;
        manager_full_address = getenv("AVA_MANAGER_ADDR");
        assert(manager_full_address != NULL && "AVA_MANAGER_ADDR is not set");
        parseServerAddress(manager_full_address, &server_info, manager_name, &manager_port);
        assert(server_info != NULL && "Unknown manager address");
        assert(manager_port > 0 && "Invalid manager port");

        /**
         * Connect manager which shall return the assigned API server's
         * address. The address can either be a full IP:port or only the port
         * if the API server is on the same machine as the manager.
         */
        int manager_fd = socket(AF_INET, SOCK_STREAM, 0);
        address.sin_family = AF_INET;
        address.sin_addr = *(struct in_addr *)server_info->h_addr;
        address.sin_port = htons(manager_port);
        fprintf(stderr, "Connect target manager (%s) at %s:%d\n",
                manager_full_address, inet_ntoa(address.sin_addr), manager_port);
        connect(manager_fd, (struct sockaddr *)&address, sizeof(address));

        struct command_base* msg = command_channel_socket_new_command(
                (struct command_channel *)chan, sizeof(struct command_base), 0);
        msg->command_type = NW_NEW_APPLICATION;
        send_socket(manager_fd, msg, sizeof(struct command_base));

        recv_socket(manager_fd, msg, sizeof(struct command_base));
        char *worker_full_address = (char *)msg->reserved_area;
        assert(worker_full_address != NULL && "No API server is assigned");
        char worker_name[128];
        int worker_port;
        struct hostent *worker_server_info;
        parseServerAddress(worker_full_address, &worker_server_info, worker_name, &worker_port);
        assert(worker_server_info != NULL && "Unknown API server address");
        assert(worker_port > 0 && "Invalid API server port");

        assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
        chan->listen_port = nw_worker_id = worker_port;
        command_channel_socket_free_command((struct command_channel *)chan, msg);
        close(manager_fd);

        /**
         * Start a TCP client to connect API server at `worker_name:worker_port`.
         */
        DEBUG_PRINT("Assigned worker at %s:%d\n", worker_name, worker_port);
        chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt_lowlatency(chan->sock_fd);
        address.sin_family = AF_INET;
        address.sin_addr = *(struct in_addr *)server_info->h_addr;
        address.sin_port = htons(worker_port);
        fprintf(stderr, "Connect target worker (%s) at %s:%d\n",
                worker_full_address, inet_ntoa(address.sin_addr), worker_port);
        connect(chan->sock_fd, (struct sockaddr *)&address, sizeof(address));
    }
    else {
        chan->listen_port = worker_port;

        /* start TCP server */
        if ((chan->listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
            perror("socket");
        }
        // Forcefully attaching socket to the worker port
        if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
            perror("setsockopt");
        }
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(chan->listen_port);

        if (bind(chan->listen_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
            perror("bind failed");
        }
        if (listen(chan->listen_fd, 10) < 0) {
            perror("listen");
        }

        fprintf(stderr, "[%d] Waiting for guestlib connection\n", chan->listen_port);
        chan->sock_fd = accept(chan->listen_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
        if (chan->sock_fd < 0) {
           perror("accept");
        }
        setsockopt_lowlatency(chan->sock_fd);

        /* Get source address */
#ifdef DEBUG
        struct sockaddr_storage source_addr;
        socklen_t source_addr_len = sizeof(struct sockaddr_storage);
        getpeername(chan->sock_fd, (struct sockaddr *)&source_addr, &source_addr_len);
        if (source_addr.ss_family == AF_INET) {
            struct sockaddr_in *s = (struct sockaddr_in *)&source_addr;
            char ipstr[64];
            inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof(ipstr));
            fprintf(stderr, "[%d] Accept guestlib at %s:%d\n", chan->listen_fd, ipstr, ntohs(s->sin_port));
        }
#endif

        /* Receive handler initialization API */
        struct command_handler_initialize_api_command init_msg;
        recv_socket(chan->sock_fd, &init_msg, sizeof(struct command_handler_initialize_api_command));
        chan->init_command_type = init_msg.new_api_id;
        chan->vm_id = init_msg.base.vm_id;
        fprintf(stderr, "[%d] Accept guestlib with API_ID=%x\n",
                chan->listen_port, chan->init_command_type);
    }

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

struct command_channel *command_channel_socket_tcp_migration_new(int worker_port, int is_source)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    chan->listen_port = worker_port + 2000;

    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    memset(&address, 0, sizeof(address));

    if (is_source) {
        /* start TCP client */
        chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);

        address.sin_family = AF_INET;
        address.sin_port = htons(chan->listen_port);
        inet_pton(AF_INET, DEST_SERVER_IP, &address.sin_addr);
        printf("connect target worker@%s:%d\n", DEST_SERVER_IP, chan->listen_port);
        connect(chan->sock_fd, (struct sockaddr *)&address, sizeof(address));
    }
    else {
        /* start TCP server */
        if ((chan->listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
            perror("socket");
        }
        // Forcefully attaching socket to the worker port
        if (setsockopt(chan->listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
            perror("setsockopt");
        }
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(chan->listen_port);

        if (bind(chan->listen_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
            perror("bind failed");
        }
        if (listen(chan->listen_fd, 10) < 0) {
            perror("listen");
        }

        printf("[target worker@%d] waiting for source worker connection\n", chan->listen_port);
        chan->sock_fd = accept(chan->listen_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);

        /* Get source address */
#ifdef DEBUG
        struct sockaddr_storage source_addr;
        socklen_t source_addr_len = sizeof(struct sockaddr_storage);
        getpeername(chan->sock_fd, (struct sockaddr *)&source_addr, &source_addr_len);
        if (source_addr.ss_family == AF_INET) {
            struct sockaddr_in *s = (struct sockaddr_in *)&source_addr;
            char ipstr[64];
            inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof(ipstr));
            printf("accept source worker@%s:%d\n", ipstr, ntohs(s->sin_port));
        }
#endif
    }

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

static struct command_channel_vtable command_channel_socket_vtable = {
  command_channel_socket_buffer_size,
  command_channel_socket_new_command,
  command_channel_socket_attach_buffer,
  command_channel_socket_send_command,
  command_channel_socket_transfer_command,
  command_channel_socket_receive_command,
  command_channel_socket_get_buffer,
  command_channel_socket_get_data_region,
  command_channel_socket_free_command,
  command_channel_socket_free,
  command_channel_socket_print_command
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the command_channel_socket_new and other socket specific APIs.
