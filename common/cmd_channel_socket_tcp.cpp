#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"
#include "common/guest_mem.h"
#include "common/cmd_handler.h"
#include "cmd_channel_socket_utilities.h"
#include "manager_service.h"

extern int nw_global_vm_id;

namespace {
  extern struct command_channel_vtable command_channel_socket_tcp_vtable;
}

struct command_channel* command_channel_min_new()
{
    struct chansocketutil::command_channel_socket* chan =
      (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_tcp_vtable);
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

    struct command_base* msg = chansocketutil::command_channel_socket_new_command((struct command_channel *)chan, sizeof(struct command_base), 0);
    msg->command_type = NW_NEW_APPLICATION;
    struct param_block_info *pb_info = (struct param_block_info *)msg->reserved_area;
    pb_info->param_local_offset = 0;
    pb_info->param_block_size = 0;
    send_socket(manager_fd, msg, sizeof(struct command_base));

    recv_socket(manager_fd, msg, sizeof(struct command_base));
    uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
    assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
    nw_worker_id = worker_port;
    chansocketutil::command_channel_socket_free_command((struct command_channel *)chan, msg);
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
    struct chansocketutil::command_channel_socket *chan = (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_tcp_vtable);
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

/**
 * TCP channel.
 * @worker_port: the listening port of worker. Only worker needs to provide
 * this argument.
 * @is_guest: 1 if it is client (guestlib), otherwise 0
 *
 * The `manager_tcp` is required to use the TCP channel.
 * */
struct command_channel* command_channel_socket_tcp_new(int worker_port, int is_guest)
{
    struct chansocketutil::command_channel_socket *chan = (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_tcp_vtable);
    pthread_mutex_init(&chan->send_mutex, NULL);
    pthread_mutex_init(&chan->recv_mutex, NULL);

    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    memset(&address, 0, sizeof(address));

    if (is_guest) {
        chan->vm_id = nw_global_vm_id = 1;

        /**
         * Get manager's host address from ENV('AVA_MANAGER_ADDR') which must
         * be a full IP:PORT (e.g. 0.0.0.0:3333).
         * Manager shall return the assigned API servers' addresses which must
         * be full IP:PORT addresses as well.
         */
        std::string manager_address(getenv("AVA_MANAGER_ADDR"));
        assert(!manager_address.empty() && "Unknown manager address");

        auto channel = grpc::CreateChannel(manager_address, grpc::InsecureChannelCredentials());
        auto client  = std::make_unique<ManagerServiceClient>(channel);
        std::vector<uint64_t> gpu_mem;
        std::vector<std::string> worker_address = client->AssignWorker(1, 0, gpu_mem);
        assert(!worker_address.empty() && "No API server is assigned");

        char worker_name[128];
        int worker_port;
        struct hostent *worker_server_info;
        parseServerAddress(worker_address[0].c_str(), &worker_server_info, worker_name, &worker_port);
        assert(worker_server_info != NULL && "Unknown API server address");
        assert(worker_port > 0 && "Invalid API server port");

        assert(nw_worker_id == 0); // TODO: Move assignment to nw_worker_id out of unrelated constructor.
        chan->listen_port = nw_worker_id = worker_port;

        /**
         * Start a TCP client to connect API server at `worker_name:worker_port`.
         */
        DEBUG_PRINT("Assigned worker at %s:%d\n", worker_name, worker_port);
        chan->sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt_lowlatency(chan->sock_fd);
        address.sin_family = AF_INET;
        address.sin_addr = *(struct in_addr *)worker_server_info->h_addr;
        address.sin_port = htons(worker_port);
        fprintf(stderr, "Connect target worker (%s) at %s:%d\n",
                worker_address[0].c_str(), inet_ntoa(address.sin_addr), worker_port);
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

struct command_channel* command_channel_socket_tcp_migration_new(int worker_port, int is_source)
{
    struct chansocketutil::command_channel_socket *chan =
      (struct chansocketutil::command_channel_socket *)malloc(sizeof(struct chansocketutil::command_channel_socket));
    command_channel_preinitialize((struct command_channel *) chan, &command_channel_socket_tcp_vtable);
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

namespace {
  struct command_channel_vtable command_channel_socket_tcp_vtable = {
    chansocketutil::command_channel_socket_buffer_size,
    chansocketutil::command_channel_socket_new_command,
    chansocketutil::command_channel_socket_attach_buffer,
    chansocketutil::command_channel_socket_send_command,
    chansocketutil::command_channel_socket_transfer_command,
    chansocketutil::command_channel_socket_receive_command,
    chansocketutil::command_channel_socket_get_buffer,
    chansocketutil::command_channel_socket_get_data_region,
    chansocketutil::command_channel_socket_free_command,
    chansocketutil::command_channel_socket_free,
    chansocketutil::command_channel_socket_print_command
  };
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the chansocketutil::command_channel_socket_new and other socket specific APIs.
