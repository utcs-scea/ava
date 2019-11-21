#include <errno.h>
#include <fcntl.h>
#include <glib.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ipc.h>
#include <sys/mman.h>

#include "common/cmd_channel_impl.h"
#include "common/cmd_handler.h"
#include "common/guest_mem.h"
#include "common/ioctl.h"
#include "common/register.h"
#include "common/socket.h"

int listen_fd;
int worker_id;
GHashTable *worker_info;

int worker_pool_enabled;

__sighandler_t original_sigint_handler = SIG_DFL;

void sigint_handler(int signo)
{
    if (listen_fd > 0)
        close(listen_fd);
    signal(signo, original_sigint_handler);
    raise(signo);
}

int main(int argc, char *argv[])
{
    /* parse environment variables */
    worker_pool_enabled = (getenv("AVA_WPOOL") && !strcmp(getenv("AVA_WPOOL"), "TRUE"));
    printf("* worker pool: %s\n", worker_pool_enabled ? "enabled" : "disabled");

    /* setup signal handler */
    if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR)
        printf("failed to catch SIGINT\n");

    /* setup worker info hash table */
    worker_info = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, free);

    /* initialize TCP socket */
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    int client_fd;
    pid_t child;
    struct command_base msg, response;
    struct param_block_info *pb_info;
    struct param_block_info *pb_hash;
    uintptr_t *worker_port;
    char str_port[10];
    char *argv_list[] = {"worker", str_port, NULL};

    worker_id = 1;
    int assigned_worker_id = 1;

    if ((listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket");
    }
    // Forcefully attaching socket to the manager port
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(WORKER_MANAGER_PORT);

    if (bind(listen_fd, (struct sockaddr *)&address, sizeof(address))<0) {
        perror("bind failed");
    }
    if (listen(listen_fd, 10) < 0) {
        perror("listen");
    }

    /* spawn worker pool */
    if (worker_pool_enabled) {
        for (; assigned_worker_id <= WORKER_POOL_SIZE; assigned_worker_id++) {
            child = fork();
            if (child == 0) {
                sprintf(str_port, "%d", assigned_worker_id + WORKER_PORT_BASE);
                goto spawn_worker;
            }
        }
    }

    /* polling new applications */
    do {
        client_fd = accept(listen_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);

        /* get guestlib info */
        recv_socket(client_fd, &msg, sizeof(struct command_base));
        pb_info = (struct param_block_info *)msg.reserved_area;
        switch (msg.command_type) {
            case NW_NEW_APPLICATION:
                pb_hash = (struct param_block_info *)malloc(sizeof(struct param_block_info));
                *pb_hash = *pb_info;
                g_hash_table_insert(worker_info, (gpointer)(uintptr_t)(worker_id + WORKER_PORT_BASE), (gpointer)pb_hash);
                break;

            case COMMAND_START_MIGRATION:
                worker_port = (uintptr_t *)msg.reserved_area;
                printf("[manager] request to migrate from worker@%lu to worker%d\n",
                        *worker_port, worker_id + WORKER_PORT_BASE);
                pb_hash = (struct param_block_info *)g_hash_table_lookup(worker_info, (gpointer)(*worker_port));

                if (!pb_hash) {
                    printf("[manager] worker_info faults\n");
                    close(client_fd);
                    exit(0);
                }
                *pb_info = *pb_hash;
                break;

            default:
                printf("[manager] wrong message type\n");
                close(client_fd);
        }

        /* return worker port to guestlib */
        response.api_id = INTERNAL_API;
        worker_port = (uintptr_t *)response.reserved_area;
        *worker_port = worker_id + WORKER_PORT_BASE;
        send_socket(client_fd, &response, sizeof(struct command_base));
        close(client_fd);

        /* spawn a worker */
        child = fork();
        if (child == 0) {
            close(listen_fd);
            break;
        }

        if (worker_pool_enabled)
            assigned_worker_id++;
        worker_id++;
    } while (1);

    if (!worker_pool_enabled)
        sprintf(str_port, "%d", worker_id + WORKER_PORT_BASE);
    else
        sprintf(str_port, "%d", assigned_worker_id + WORKER_PORT_BASE);

    /* spawn worker */
spawn_worker:
    printf("[manager] spawn new worker port=%s\n", str_port);
    if (execv("worker", argv_list) < 0) {
        perror("execv worker");
    }

    return 0;
}
