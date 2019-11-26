#include <errno.h>
#include <fcntl.h>
#include <glib.h>
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

static int kvm_fd;
static int policy_id;
static struct sched_policy sched_policy = {
    .module_name           = "ava_policy_device_time_hp",
    .cb_struct             = "device_time_hp_func",
    .consume_func_name     = "consume_vm_device_time_hp",
};

static int worker_pool_enabled;
static int policy_enabled;

__sighandler_t original_sigint_handler = SIG_DFL;

void sigint_handler(int signo)
{
    if (listen_fd > 0)
        close(listen_fd);

    if (policy_enabled && policy_id > 0) {
        if (ioctl(kvm_fd, KVM_REMOVE_SCHEDULING_POLICY, policy_id) < 0) {
            printf("\nFailed to uninstall policy: %s\n", sched_policy.module_name);
        }
        else {
            printf("\n* Uninstall policy: %s\n", sched_policy.module_name);
        }
        close(kvm_fd);
    }

    signal(signo, original_sigint_handler);
    raise(signo);
}

int main(int argc, char *argv[])
{
    /* Parse arguments */
    int i;
    if (argc > 1) {
        /* Help menu */
        for (i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                fprintf(stderr, "%s [--help] [--policy]\n", argv[0]);
                return 0;
            }
        }

        for (i = 1; i < argc; i++)
            if (strcmp(argv[i], "--policy") == 0 || strcmp(argv[i], "-p") == 0) {
                printf("* Install policy: %s\n", sched_policy.module_name);
                policy_enabled = 1;
            }
    }

    /* parse environment variables */
    worker_pool_enabled = (getenv("AVA_WPOOL") && !strcmp(getenv("AVA_WPOOL"), "TRUE"));
    printf("* API server pool: %s\n", worker_pool_enabled ? "enabled" : "disabled");

    /* setup signal handler */
    if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR)
        printf("Failed to catch SIGINT\n");

    /* setup worker info hash table */
    worker_info = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, free);

    /* setup scheduling policy */
    if (policy_enabled) {
        sched_policy.module_name_len = strlen(sched_policy.module_name) + 1;
        sched_policy.cb_struct_len = strlen(sched_policy.cb_struct) + 1;
        sched_policy.consume_func_name_len = strlen(sched_policy.consume_func_name) + 1;

        if ((kvm_fd = open("/dev/kvm-vgpu", O_RDWR | O_NONBLOCK)) < 0) {
            fprintf(stderr, "Failed to open /dev/kvm-vgpu\n");
            exit(0);
        }
        if ((policy_id = ioctl(kvm_fd, KVM_SET_SCHEDULING_POLICY, (unsigned long)&sched_policy)) <= 0) {
            fprintf(stderr, "Resource policy is not found\n");
            exit(0);
        }
        else {
            printf("* Assigned policy ID %d\n", policy_id);
        }
    }

    /* initialize vsock */
    struct sockaddr_vm sa_listen;
    int client_fd;
    pid_t child;
    struct command_base msg, response;
    struct param_block_info *pb_info;
    struct param_block_info *pb_hash;
    uintptr_t *worker_port;
    int guest_cid;
    char str_port[10];
    char *argv_list[] = {"worker", str_port, NULL};

    worker_id = 1;
    int assigned_worker_id = 1;

    listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, WORKER_MANAGER_PORT);
    listen_vm_socket(listen_fd, &sa_listen);

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
        client_fd = accept_vm_socket(listen_fd, &guest_cid);

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
                printf("[manager] Request to migrate from worker@%lu to worker%d\n",
                        *worker_port, worker_id + WORKER_PORT_BASE);
                pb_hash = (struct param_block_info *)g_hash_table_lookup(worker_info, GINT_TO_POINTER(*worker_port));

                if (!pb_hash) {
                    printf("[manager] Worker_info faults\n");
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
    printf("[manager] Spawn new worker port=%s\n", str_port);
    if (execv("./worker", argv_list) < 0) {
        perror("execv worker");
    }

    return 0;
}
